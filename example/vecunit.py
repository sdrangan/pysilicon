from pysilicon.hwobj import HwObj, MemIF
import numpy as np

class ControlCmd(Packet):
    """
    Control command packet for the vector unit.  The command
    is used to start execution of a program or to reset the vector unit.
    It is sent in the control FIFO interface.

    It derives from Packet so that it can be serialized on 
    the FIFO interface
    """
    def __init__(self):
        self.add_field('opcode', EnumType('start', 'reset'),
                       desc='Control opcode. Start or ' \
                       'reset the vector unit.')
        self.add_field('program_addr', IntType(32, signed=False),
                       desc='Address of the program to execute. Only used for start opcode.')


class VecUnit(HwObj):
    """
    Vector Unit hardware object.

    A simple SIMD-type vector unit that supports a small set of
    instructions

    """
    def __init__(
            self,
            name: str,
            nreg : int = 8,
            npara : int = 1):
        """
        Initialize the vector unit.

        Parameters
        ----------
        name : str
            Name of the vector unit.
        nreg : int
            Number of vector registers all assumed to be float32.  
        npara : int
            Number of parallel operations the vector unit can perform.
        """
        super().__init__(name)
        self.nreg = nreg
        self.npara = npara

        # Add the master specifications for the memory interfaces.
        # We have memory interfaces for the program and data.
        # The data interface is read and write, while the program interface
        # is read only.
        data_if_spec  = MemIF.master_spec(addr_width=32, data_width=32, 
                                     supported_ops=['read', 'write'])
        prog_if_spec  = MemIF.master_spec(addr_width=32, data_width=32,
                                     supported_ops=['read'])
        self.add_master_interface('data_mem', data_if_spec)
        self.add_master_interface('prog_mem', prog_if_spec)

        # Create the regiter file which is modeled simply as an array
        # of float32 values. 
        self.register_file = np.zeros((nreg,), dtype=np.float32)
        self.pc = 0  # Program counter

        # Add a slave control interface.  
        ctrl_fifo = AxiStreamIF(
            name='ctrl_fifo', data_width=32, depth=16,
            action=self.process_ctrl_cmd)
        self.add_slave_interface(ctrl_fifo)

    def process_ctrl_cmd(self, cmd: ControlCmd):
        """
        Process a control command received on the control FIFO interface.
        This method will be called when a command is received on the control
        FIFO interface. The command will be deserialized into a ControlCmd
        object and passed to this method. The method should implement the
        logic to handle the control command, such as starting execution of a
        program or resetting the vector unit. Note that this method is not
        called directly, but is registered as an action for the control FIFO
        interface, so that it will be called automatically when a command is
        received on the interface.
        """
        if cmd.opcode == 'start':
            self.start_execution(cmd.program_addr)
        elif cmd.opcode == 'reset':
            self.reset()
        else:
            raise ValueError(f"Unknown opcode {cmd.opcode} in control command.")