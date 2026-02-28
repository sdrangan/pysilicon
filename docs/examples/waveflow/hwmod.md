# Vector Unit 

```python
from pysilicon import HwObj, Code

class VectorUnit(HwObj):
    def __init__(
        self, 
        nreg : , width=32):

        # Registers
        self.reg = []
        for i in range(nreg):
            reg.append( Register(np.float32) )

        # Program counter
        self.pc = 0

        # Interfaces
        pmem_if = MemIf( width)
        self.add_interface()


        self.code: Code = Code([])
        self.mem = mem  # abstract memory interface

    def step(self):
        instr = self.code[self.pc]

        if isinstance(instr, VLoad):
            self.regs[instr.dst] = self.mem.read(instr.addr)
            self.pc += 1

        elif isinstance(instr, VStore):
            self.mem.write(instr.addr, self.regs[instr.src])
            self.pc += 1

        elif isinstance(instr, VAdd):
            self.regs[instr.dst] = (
                self.regs[instr.src1] + self.regs[instr.src2]
            )
            self.pc += 1

        elif isinstance(instr, VMul):
            self.regs[instr.dst] = (
                self.regs[instr.src1] * self.regs[instr.src2]
            )
            self.pc += 1

        elif isinstance(instr, VRelu):
            v = self.regs[instr.src]
            self.regs[instr.dst] = v if v > 0 else 0
            self.pc += 1

        elif isinstance(instr, VLoop):
            if instr.count > 0:
                instr.count -= 1
                self.pc = instr.target
            else:
                self.pc += 1
```