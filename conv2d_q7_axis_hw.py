from pynq import Overlay
import time
import numpy as np

# Try modern allocate() first, fall back to Xlnk on older PYNQ
try:
    from pynq import allocate
    _alloc_mode = "allocate"
except ImportError:
    from pynq import Xlnk
    _xlnk = Xlnk()
    _alloc_mode = "xlnk"


class Conv2DQ7AxisAccelerator:
    """
    AXI-Stream conv2d_q7 accelerator:

    Input stream (S_AXIS, int16 words):
        - 9 kernel samples (Q7 int8 in low 8 bits)
        - 784 image samples (28x28, Q7 int8 in low 8 bits)
      Total 793 words, TLAST on last word.

    Output stream (M_AXIS, int16 words):
        - 676 outputs (26x26, Q14 int16)
        - TLAST on last word.
    """

    IMG_H = 28
    IMG_W = 28
    OUT_H = 26
    OUT_W = 26

    STREAM_IN_WORDS  = 9 + IMG_H * IMG_W   # 793
    STREAM_OUT_WORDS = OUT_H * OUT_W       # 676

    def __init__(self, bitfile="conv2d_q7_axis.bit"):
        # Load overlay
        self.overlay = Overlay(bitfile)

        # --- Auto-detect AXI DMA IP ---
        dma_ip_name = None
        for name, desc in self.overlay.ip_dict.items():
            # desc['type'] e.g. "xilinx.com:ip:axi_dma:7.1"
            if 'axi_dma_0' in desc.get('type', ''):
                dma_ip_name = name
                break

        if dma_ip_name is None:
            raise RuntimeError(
                "No AXI DMA IP found in overlay '{}'. "
                "Check Vivado block design: add an AXI DMA and regenerate "
                "the bitstream.".format(bitfile)
            )

        self.dma = getattr(self.overlay, dma_ip_name)

        # Allocate physically contiguous buffers
        if _alloc_mode == "allocate":
            self.in_buf  = allocate(shape=(self.STREAM_IN_WORDS,),  dtype=np.int16)
            self.out_buf = allocate(shape=(self.STREAM_OUT_WORDS,), dtype=np.int16)
        else:
            self.in_buf  = _xlnk.cma_array(shape=(self.STREAM_IN_WORDS,),  dtype=np.int16)
            self.out_buf = _xlnk.cma_array(shape=(self.STREAM_OUT_WORDS,), dtype=np.int16)

    @staticmethod
    def _to_q7_int16(val):
        """Clamp to [-128,127] and store in int16; low 8 bits are Q7 value."""
        v = int(val)
        if v < -128:
            v = -128
        elif v > 127:
            v = 127
        return np.int16(v)

    def _prepare_input_stream(self, image, kernel):
        """
        Fill self.in_buf with:
        [ 9 kernel values (row-major), 784 image values (row-major) ]
        Each as int16 with low 8 bits = Q7.
        """
        idx = 0

        # Kernel 3x3
        for r in range(3):
            for c in range(3):
                self.in_buf[idx] = self._to_q7_int16(kernel[r][c])
                idx += 1

        # Image 28x28
        for r in range(self.IMG_H):
            row = image[r]
            for c in range(self.IMG_W):
                self.in_buf[idx] = self._to_q7_int16(row[c])
                idx += 1

        assert idx == self.STREAM_IN_WORDS

    def _wait_channel(self, ch, timeout_s=1.0, label="send"):
        """Wait for a DMA channel with timeout; raise clear error if stuck."""
        t0 = time.time()
        while True:
            if ch.idle:
                return
            if time.time() - t0 > timeout_s:
                # Dump status register for debugging
                status = ch._mmio.read(ch._offset + 4)
                raise RuntimeError(
                    "DMA {}channel did not go idle (timeout). "
                    "Status=0x{:08X}. This usually means the AXI-Stream "
                    "handshake is broken: TREADY/TVALID/TLAST or DMA config."
                    .format(label, status)
                )

    def conv2d_q7(self, image, kernel):
        """
        Run hardware accelerator:
          image  : 28x28 list-of-lists of ints (Q7)
          kernel : 3x3  list-of-lists of ints (Q7)
        Returns:
          26x26 list-of-lists of ints (Q14).
        """
        self._prepare_input_stream(image, kernel)

        # Start RX first (S2MM), then TX (MM2S)
        self.dma.recvchannel.transfer(self.out_buf)
        self.dma.sendchannel.transfer(self.in_buf)

        # Wait with timeout instead of infinite busy-wait
        self._wait_channel(self.dma.sendchannel, timeout_s=2.0, label="MM2S ")
        self._wait_channel(self.dma.recvchannel, timeout_s=2.0, label="S2MM ")

        # Convert output buffer to Python list
        out = [[0] * self.OUT_W for _ in range(self.OUT_H)]
        idx = 0
        for i in range(self.OUT_H):
            for j in range(self.OUT_W):
                v = int(self.out_buf[idx])   # int16 -> Python int
                out[i][j] = v
                idx += 1
        return out


_accel = None

def conv2d_q7(image, kernel, bitfile="conv2d_q7_axis.bit"):
    """
    Drop-in replacement for the software conv2d_q7(image, kernel)
    from your discrete CNN lab (Q7 inputs, Q14 outputs). :contentReference[oaicite:0]{index=0}
    """
    global _accel
    if _accel is None:
        _accel = Conv2DQ7AxisAccelerator(bitfile)
    return _accel.conv2d_q7(image, kernel)
