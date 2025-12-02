----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 12/02/2025 09:46:37 AM
-- Design Name: 
-- Module Name: conv2d_q7_axis - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;

entity conv2d_q7_axis is
  generic (
    C_S_AXIS_TDATA_WIDTH : integer := 16; -- input stream width
    C_M_AXIS_TDATA_WIDTH : integer := 16  -- output stream width
  );
  port (
    -------------------------------------------------------------------
    -- Global clock & reset
    -------------------------------------------------------------------
    aclk    : in  std_logic;
    aresetn : in  std_logic;

    -------------------------------------------------------------------
    -- AXI4-Stream slave (input: kernel + image)
    -------------------------------------------------------------------
    s_axis_tdata : in  std_logic_vector(C_S_AXIS_TDATA_WIDTH-1 downto 0);
    s_axis_tvalid: in  std_logic;
    s_axis_tready: out std_logic;
    s_axis_tlast : in  std_logic;

    -------------------------------------------------------------------
    -- AXI4-Stream master (output: 26x26 feature map)
    -------------------------------------------------------------------
    m_axis_tdata : out std_logic_vector(C_M_AXIS_TDATA_WIDTH-1 downto 0);
    m_axis_tvalid: out std_logic;
    m_axis_tready: in  std_logic;
    m_axis_tlast : out std_logic
  );
end entity conv2d_q7_axis;

architecture rtl of conv2d_q7_axis is

  -- Sizes
  constant IMG_W : integer := 28;
  constant IMG_H : integer := 28;
  constant OUT_W : integer := 26;
  constant OUT_H : integer := 26;

  constant NUM_KERNEL : integer := 9;                -- 3x3
  constant NUM_PIXELS : integer := IMG_W * IMG_H;    -- 784
  constant TOTAL_INPUT_WORDS : integer := NUM_KERNEL + NUM_PIXELS; -- 793

  -- Memories (Q7)
  type img_mem_t is array (0 to IMG_H-1, 0 to IMG_W-1) of signed(7 downto 0);
  type ker_mem_t is array (0 to 2, 0 to 2) of signed(7 downto 0);

  signal img_mem : img_mem_t;
  signal ker_mem : ker_mem_t;

  -- FSM
  type state_t is (LOAD, COMPUTE, OUTPUT);
  signal state : state_t := LOAD;

  signal in_count : integer range 0 to TOTAL_INPUT_WORDS-1 := 0;

  -- Stream regs
  signal s_axis_tready_reg : std_logic := '0';
  signal m_axis_tvalid_reg : std_logic := '0';
  signal m_axis_tdata_reg  : std_logic_vector(C_M_AXIS_TDATA_WIDTH-1 downto 0)
                            := (others => '0');
  signal m_axis_tlast_reg  : std_logic := '0';

  -- Convolution indices
  signal conv_i : integer range 0 to OUT_H-1 := 0;
  signal conv_j : integer range 0 to OUT_W-1 := 0;
  signal tap    : integer range 0 to 9       := 0;
  signal sum_acc: integer range -262144 to 262143 := 0;  -- enough for 9*Q7*Q7

  -- Saturation helper
  function sat16 (x : integer) return signed is
    variable tmp : integer := x;
  begin
    if tmp > 32767 then
      tmp := 32767;
    elsif tmp < -32768 then
      tmp := -32768;
    end if;
    return to_signed(tmp, 16);
  end function;

begin

  -- Connect regs to ports
  s_axis_tready <= s_axis_tready_reg;
  m_axis_tvalid <= m_axis_tvalid_reg;
  m_axis_tdata  <= m_axis_tdata_reg;
  m_axis_tlast  <= m_axis_tlast_reg;

  --------------------------------------------------------------------
  -- Main sequential process: LOAD -> COMPUTE -> OUTPUT
  --------------------------------------------------------------------
  process (aclk)
    variable pix     : signed(7 downto 0);
    variable ker     : signed(7 downto 0);
    variable row_off : integer range 0 to 2;
    variable col_off : integer range 0 to 2;
    variable prod    : integer range -16384 to 16383;
    variable idx     : integer range 0 to NUM_PIXELS-1;
  begin
    if rising_edge(aclk) then
      if aresetn = '0' then
        ----------------------------------------------------------------
        -- Reset
        ----------------------------------------------------------------
        state             <= LOAD;
        in_count          <= 0;
        s_axis_tready_reg <= '1';
        m_axis_tvalid_reg <= '0';
        m_axis_tdata_reg  <= (others => '0');
        m_axis_tlast_reg  <= '0';
        conv_i            <= 0;
        conv_j            <= 0;
        tap               <= 0;
        sum_acc           <= 0;

      else
        case state is

          ----------------------------------------------------------------
          -- LOAD: receive 9 kernel + 784 image samples via S_AXIS
          ----------------------------------------------------------------
          when LOAD =>
            s_axis_tready_reg <= '1';    -- ready to accept stream
            m_axis_tvalid_reg <= '0';
            m_axis_tlast_reg  <= '0';

            if s_axis_tvalid = '1' and s_axis_tready_reg = '1' then
              -- Grab Q7 from low 8 bits
              pix := signed(s_axis_tdata(7 downto 0));

              if in_count < NUM_KERNEL then
                -- First 9 words -> kernel
                ker_mem(in_count / 3, in_count mod 3) <= pix;
              else
                -- Remaining 784 words -> image
                idx := in_count - NUM_KERNEL;     -- 0..783
                img_mem(idx / IMG_W, idx mod IMG_W) <= pix;
              end if;

              if in_count = TOTAL_INPUT_WORDS-1 then
                -- Last input word received
                in_count          <= 0;
                s_axis_tready_reg <= '0';
                state             <= COMPUTE;
                conv_i            <= 0;
                conv_j            <= 0;
                tap               <= 0;
                sum_acc           <= 0;
              else
                in_count <= in_count + 1;
              end if;
            end if;

          ----------------------------------------------------------------
          -- COMPUTE: perform 3x3 MAC for current (conv_i, conv_j)
          ----------------------------------------------------------------
          when COMPUTE =>
            s_axis_tready_reg <= '0';
            m_axis_tvalid_reg <= '0';
            m_axis_tlast_reg  <= '0';

            if tap < 9 then
              -- Determine (row_off, col_off) for current tap
              case tap is
                when 0 => row_off := 0; col_off := 0;
                when 1 => row_off := 0; col_off := 1;
                when 2 => row_off := 0; col_off := 2;
                when 3 => row_off := 1; col_off := 0;
                when 4 => row_off := 1; col_off := 1;
                when 5 => row_off := 1; col_off := 2;
                when 6 => row_off := 2; col_off := 0;
                when 7 => row_off := 2; col_off := 1;
                when others => row_off := 2; col_off := 2;
              end case;

              pix  := img_mem(conv_i + row_off, conv_j + col_off);
              ker  := ker_mem(row_off, col_off);
              prod := to_integer(pix) * to_integer(ker);

              sum_acc <= sum_acc + prod;
              tap     <= tap + 1;
            else
              -- Finished 9 taps -> go send this output
              state <= OUTPUT;
            end if;

          ----------------------------------------------------------------
          -- OUTPUT: stream current sum_acc as one Q14 sample via M_AXIS
          ----------------------------------------------------------------
          when OUTPUT =>
            s_axis_tready_reg <= '0';

            if m_axis_tvalid_reg = '0' then
              -- Present output value on TDATA
              m_axis_tdata_reg <= std_logic_vector(sat16(sum_acc));

              if conv_i = OUT_H-1 and conv_j = OUT_W-1 then
                m_axis_tlast_reg <= '1';   -- last output in frame
              else
                m_axis_tlast_reg <= '0';
              end if;

              m_axis_tvalid_reg <= '1';

            else
              -- Wait for consumer to accept the data
              if m_axis_tvalid_reg = '1' and m_axis_tready = '1' then
                -- Output accepted
                m_axis_tvalid_reg <= '0';
                m_axis_tlast_reg  <= '0';
                sum_acc           <= 0;
                tap               <= 0;

                if conv_j < OUT_W-1 then
                  -- Next column in same row
                  conv_j <= conv_j + 1;
                  state  <= COMPUTE;

                else
                  -- End of row; move to next row or finish frame
                  conv_j <= 0;
                  if conv_i < OUT_H-1 then
                    conv_i <= conv_i + 1;
                    state  <= COMPUTE;
                  else
                    -- Entire 26x26 frame done
                    conv_i            <= 0;
                    state             <= LOAD;
                    s_axis_tready_reg <= '1';
                  end if;
                end if;
              end if;
            end if;

        end case;
      end if;
    end if;
  end process;

end architecture rtl;
