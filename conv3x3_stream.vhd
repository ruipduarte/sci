-- conv3x3_stream.vhd
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity conv3x3_stream is
  generic (
    DATA_IN_WIDTH  : integer := 8;   -- input pixel width (signed)
    DATA_OUT_WIDTH : integer := 16   -- output accumulator width (signed)
  );
  port (
    -- AXI4-Stream Slave (input pixels)
    s_axis_tdata  : in  std_logic_vector(DATA_IN_WIDTH-1 downto 0);
    s_axis_tvalid : in  std_logic;
    s_axis_tready : out std_logic;
    s_axis_tlast  : in  std_logic;

    -- AXI4-Stream Master (output conv samples)
    m_axis_tdata  : out std_logic_vector(DATA_OUT_WIDTH-1 downto 0);
    m_axis_tvalid : out std_logic;
    m_axis_tready : in  std_logic;
    m_axis_tlast  : out std_logic;

    -- Kernel interface (simple register interface)
    ker00, ker01, ker02 : in std_logic_vector(DATA_IN_WIDTH-1 downto 0);
    ker10, ker11, ker12 : in std_logic_vector(DATA_IN_WIDTH-1 downto 0);
    ker20, ker21, ker22 : in std_logic_vector(DATA_IN_WIDTH-1 downto 0);

    clk   : in std_logic;
    rst   : in std_logic
  );
end entity;

architecture rtl of conv3x3_stream is
  constant IMG_W : integer := 28;
  constant IMG_H : integer := 28;
  constant OUT_W : integer := IMG_W - 2;
  constant OUT_H : integer := IMG_H - 2;

  -- line buffers: simple shift registers of length IMG_W
  type line_t is array (0 to IMG_W-1) of signed(DATA_IN_WIDTH-1 downto 0);
  signal line0, line1, line2 : line_t;
  signal col_ptr : integer range 0 to IMG_W-1 := 0;
  signal row_ptr : integer range 0 to IMG_H-1 := 0;

  signal valid_win : std_logic := '0';
  -- registers to hold latest 3x3 window values
  signal w00,w01,w02,w10,w11,w12,w20,w21,w22 : signed(DATA_IN_WIDTH-1 downto 0);

  -- kernel as signed
  signal k00,k01,k02,k10,k11,k12,k20,k21,k22 : signed(DATA_IN_WIDTH-1 downto 0);

  signal out_sample : signed(DATA_OUT_WIDTH-1 downto 0);
  signal out_valid  : std_logic := '0';
  signal out_last   : std_logic := '0';
begin

  -- kernel assignments (combinational)
  k00 <= signed(ker00); k01 <= signed(ker01); k02 <= signed(ker02);
  k10 <= signed(ker10); k11 <= signed(ker11); k12 <= signed(ker12);
  k20 <= signed(ker20); k21 <= signed(ker21); k22 <= signed(ker22);

  -- input acceptance
  s_axis_tready <= '1';  -- always ready (simple flow control)

  conv_proc: process(clk)
    variable sum : integer; -- use integer accumulator large enough
  begin
    if rising_edge(clk) then
      if rst = '1' then
        col_ptr <= 0;
        row_ptr <= 0;
        valid_win <= '0';
        out_valid <= '0';
        out_last <= '0';
      else
        if s_axis_tvalid = '1' then
          -- shift line buffers and insert new pixel into line2 end
          line0(col_ptr) <= line1(col_ptr);
          line1(col_ptr) <= line2(col_ptr);
          line2(col_ptr) <= signed(s_axis_tdata);

          -- update window registers when possible
          if col_ptr >= 2 and row_ptr >= 2 then
            -- map indices (using col_ptr to index into three lines)
            w00 <= line0(col_ptr-2);
            w01 <= line0(col_ptr-1);
            w02 <= line0(col_ptr);
            w10 <= line1(col_ptr-2);
            w11 <= line1(col_ptr-1);
            w12 <= line1(col_ptr);
            w20 <= line2(col_ptr-2);
            w21 <= line2(col_ptr-1);
            w22 <= line2(col_ptr);
            valid_win <= '1';
          else
            valid_win <= '0';
          end if;

          -- compute output only when window valid
          if valid_win = '1' then
            -- integer accumulation: convert signed to integer and multiply
            sum := to_integer(w00) * to_integer(k00)
                 + to_integer(w01) * to_integer(k01)
                 + to_integer(w02) * to_integer(k02)
                 + to_integer(w10) * to_integer(k10)
                 + to_integer(w11) * to_integer(k11)
                 + to_integer(w12) * to_integer(k12)
                 + to_integer(w20) * to_integer(k20)
                 + to_integer(w21) * to_integer(k21)
                 + to_integer(w22) * to_integer(k22);
            -- saturate to DATA_OUT_WIDTH
            if sum > 2**(DATA_OUT_WIDTH-1)-1 then
              out_sample <= to_signed(2**(DATA_OUT_WIDTH-1)-1, DATA_OUT_WIDTH);
            elsif sum < -2**(DATA_OUT_WIDTH-1) then
              out_sample <= to_signed(-2**(DATA_OUT_WIDTH-1), DATA_OUT_WIDTH);
            else
              out_sample <= to_signed(sum, DATA_OUT_WIDTH);
            end if;
            out_valid <= '1';
            -- tlast: assert at end of an output row
            if col_ptr = IMG_W-1 and row_ptr >= 2 then
              -- when finishing last input of row producing last conv sample of that row:
              if col_ptr = IMG_W-1 and (col_ptr-2) = OUT_W-1 then
                out_last <= '1';
              else
                out_last <= '0';
              end if;
            else
              out_last <= '0';
            end if;
          else
            out_valid <= '0';
            out_last <= '0';
          end if;

          -- update column pointer
          if col_ptr = IMG_W-1 then
            col_ptr <= 0;
            if row_ptr = IMG_H-1 then
              row_ptr <= 0;
            else
              row_ptr <= row_ptr + 1;
            end if;
          else
            col_ptr <= col_ptr + 1;
          end if;
        else
          out_valid <= '0';
          out_last <= '0';
        end if;
      end if;
    end if;
  end process;

  -- assign stream outputs
  m_axis_tdata  <= std_logic_vector(out_sample);
  m_axis_tvalid <= out_valid;
  m_axis_tlast  <= out_last;

end architecture;
