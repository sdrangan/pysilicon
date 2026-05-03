# SchemaTransferIF — Design Plan (revised)

## 1. Overview

`SchemaTransferIF` is a logical interface for transmitting and receiving serializable
objects over physical transport mechanisms such as `StreamIF` and `AXIMMCrossBarIF`.

The interface is **agnostic to framing**: it calls `obj.serialize(word_bw)` on the
transmit side and `schema_type().deserialize(words, word_bw)` on the receive side.
Whether `schema_type` is a `DataSchema` subclass (single known type, no header) or a
`DataUnion` subclass (multi-type dispatch via header) is entirely the caller's concern.

```
Application layer:  Component.write(obj)          rx_proc(obj)
                          |                             |
Logical layer:   SchemaTransferIFMaster     SchemaTransferIFSlave
                          |                             |
Transport layer:     PhysicalTransport  (StreamTransport | AXIMMTransport | ...)
                          |                             |
Physical layer:  StreamIFMaster / AXIMMCrossBarIFMaster   StreamIFSlave / ...
```

---

## 2. Goals

1. Allow a `Component` to transmit any serializable object through a single `write(obj)`
   call, regardless of the underlying physical interface.
2. Allow a `Component` to receive deserialized objects via an `rx_proc(obj)` callback
   or a `simpy.Store` queue.
3. Support both single-type and multi-type protocols by choosing the right `schema_type`
   at construction time — `SchemaTransferIF` itself has no opinion on framing.
4. Decouple the logical interface from the physical layer through a `PhysicalTransport`
   abstraction.

---

## 3. Non-Goals / Out of Scope (Initial Version)

- Flow control or acknowledgment beyond what the underlying physical layer provides.
- Schema versioning or migration.
- HLS code generation for `SchemaTransferIF` endpoints.
- Streaming fragmentation for objects larger than the physical layer's maximum burst.

---

## 4. Wire Protocol

`SchemaTransferIF` imposes no wire format of its own. The format is determined
entirely by the `schema_type` passed at construction:

| `schema_type` | Wire format | Header? |
|---|---|---|
| `type[DataSchema]` subclass | `schema.serialize(word_bw)` — payload only | No |
| `type[DataUnion]` subclass | `DataUnionHdr` + padded payload | Yes (inside DataUnion) |

In both cases `schema_type.nwords_per_inst(word_bw)` gives the fixed burst size,
which the transport layer uses to frame bursts.

---

## 5. Schema Registry and DataUnion (already implemented)

`SchemaRegistry`, `register_schema`, `SchemaIDField`, `LengthField`, `DataUnionHdr`,
and `DataUnion` are all implemented in `pysilicon/hw/dataunion.py` and exported from
`pysilicon/hw/__init__.py`. There is **no** default registry singleton; every design
creates its own.

Single-type transfers require no registry at all.

---

## 6. Physical Transport Abstraction

`PhysicalTransport` is an ABC with two responsibilities: sending a word burst and
registering a receive callback.

```python
class PhysicalTransport(ABC):

    @abstractmethod
    def write_words(self, words: Words) -> ProcessGen:
        """Transmit a word burst through the physical endpoint."""
        ...

    @abstractmethod
    def set_rx_callback(self, callback: Callable[[Words], ProcessGen]) -> None:
        """Register the callback invoked when a word burst arrives."""
        ...
```

### 6.1 StreamTransport

```python
class StreamTransport(PhysicalTransport):
    master_ep: StreamIFMaster | None = None
    slave_ep: StreamIFSlave | None = None

    def write_words(self, words: Words) -> ProcessGen:
        yield from self.master_ep.write(words)

    def set_rx_callback(self, callback: Callable[[Words], ProcessGen]) -> None:
        self.slave_ep.rx_proc = callback
```

### 6.2 AXIMMTransport

```python
class AXIMMTransport(PhysicalTransport):
    master_ep: AXIMMCrossBarIFMaster | None = None
    slave_ep: AXIMMCrossBarIFSlave | None = None
    base_addr: int = 0x0000

    def write_words(self, words: Words) -> ProcessGen:
        yield from self.master_ep.write(words, self.base_addr)

    def set_rx_callback(self, callback: Callable[[Words], ProcessGen]) -> None:
        self.slave_ep.rx_write_proc = lambda words, addr: callback(words)
```

### 6.3 CrossBarTransport

```python
class CrossBarTransport(PhysicalTransport):
    input_ep: CrossBarIFInput | None = None
    output_ep: CrossBarIFOutput | None = None

    def write_words(self, words: Words) -> ProcessGen:
        yield from self.input_ep.write(words)

    def set_rx_callback(self, callback: Callable[[Words], ProcessGen]) -> None:
        self.output_ep.rx_proc = callback
```

---

## 7. SchemaTransferIF Endpoints

### 7.1 SchemaTransferIFMaster

The master has no `schema_type`: it simply serializes whatever object it is given.

```python
@dataclass
class SchemaTransferIFMaster(InterfaceEndpoint):
    transport: PhysicalTransport
    bitwidth: int = 32

    def write(self, obj) -> ProcessGen:
        yield from self.transport.write_words(obj.serialize(word_bw=self.bitwidth))
```

### 7.2 SchemaTransferIFSlave

The slave holds `schema_type` only to construct the right instance for
deserialization. There is no branch on `DataUnion` vs `DataSchema`.

```python
@dataclass
class SchemaTransferIFSlave(InterfaceEndpoint):
    transport: PhysicalTransport
    schema_type: type   # type[DataSchema] | type[DataUnion]
    bitwidth: int = 32
    rx_proc: Callable[[Any], ProcessGen] | None = None

    def pre_sim(self) -> None:
        self.queue = simpy.Store(self.sim.env)
        self.transport.set_rx_callback(self._on_words_received)

    def _on_words_received(self, words: Words) -> ProcessGen:
        obj = self.schema_type().deserialize(words, word_bw=self.bitwidth)
        yield self.queue.put(obj)
        if self.rx_proc is not None:
            yield from self.rx_proc(obj)
```

`queue` provides a `simpy.Store` for coroutines that prefer to `yield` on an
incoming object rather than use a callback.

### 7.3 SchemaTransferIF

```python
class SchemaTransferIF(Interface):
    endpoint_names = ('master', 'slave')

    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        # Validate endpoint is SchemaTransferIFMaster or SchemaTransferIFSlave
        # Validate bitwidth consistency between master and slave
        super().bind(ep_name, endpoint)
```

---

## 8. Usage Examples

### 8.1 Single-type transfer

One component always sends `SensorPacket`; the other always receives it.
No registry, no header.

```python
U8  = IntField.specialize(bitwidth=8,  signed=False)
S16 = IntField.specialize(bitwidth=16, signed=True)

class SensorPacket(DataList):
    elements = {"temp_raw": S16, "sensor_id": U8}


class TxComponent(Component):
    def __init__(self, sim, clk):
        self.stream_master = StreamIFMaster(sim=sim, bitwidth=32)
        self.schema_master = SchemaTransferIFMaster(
            sim=sim,
            transport=StreamTransport(master_ep=self.stream_master),
            bitwidth=32,
        )

    async def run_proc(self):
        yield from self.schema_master.write(SensorPacket(temp_raw=-42, sensor_id=7))


class RxComponent(Component):
    def __init__(self, sim, clk):
        self.stream_slave = StreamIFSlave(sim=sim, bitwidth=32)
        self.schema_slave = SchemaTransferIFSlave(
            sim=sim,
            transport=StreamTransport(slave_ep=self.stream_slave),
            schema_type=SensorPacket,
            bitwidth=32,
            rx_proc=self._on_packet,
        )

    def _on_packet(self, pkt: SensorPacket) -> ProcessGen:
        print(f"temp_raw={pkt.temp_raw}  sensor_id={pkt.sensor_id}")
        yield self.timeout(0)
```

Wire footprint: `SensorPacket.nwords_per_inst(32) = 1` word per transfer.

---

### 8.2 Multi-type transfer (DataUnion)

Multiple packet types share one interface. The `DataUnion` header carries `schema_id`
so the slave can dispatch; `SchemaTransferIF` itself sees only words.

```python
U8  = IntField.specialize(bitwidth=8,  signed=False)
S16 = IntField.specialize(bitwidth=16, signed=True)
U16 = IntField.specialize(bitwidth=16, signed=False)

sensor_reg = SchemaRegistry("Sensor")

@register_schema(schema_id=1, registry=sensor_reg)
class TempPacket(DataList):
    elements = {"temp_raw": S16, "sensor_id": U8}

@register_schema(schema_id=2, registry=sensor_reg)
class PressPacket(DataList):
    elements = {"pressure_pa": U16, "sensor_id": U8}

@register_schema(schema_id=3, registry=sensor_reg)
class AccelPacket(DataList):
    elements = {"ax": S16, "ay": S16, "az": S16}

SensorSchemaID = SchemaIDField.specialize(registry=sensor_reg, bitwidth=16)
SensorHdr      = DataUnionHdr.specialize(schema_id_type=SensorSchemaID)
SensorDU       = DataUnion.specialize(hdr_type=SensorHdr)


class TxComponent(Component):
    def __init__(self, sim, clk):
        self.stream_master = StreamIFMaster(sim=sim, bitwidth=32)
        self.schema_master = SchemaTransferIFMaster(
            sim=sim,
            transport=StreamTransport(master_ep=self.stream_master),
            bitwidth=32,
        )

    async def run_proc(self):
        for payload in [
            TempPacket(temp_raw=-42, sensor_id=7),
            AccelPacket(ax=100, ay=-200, az=980),
            PressPacket(pressure_pa=10132, sensor_id=3),
        ]:
            du = SensorDU()
            du.payload = payload
            yield from self.schema_master.write(du)


class RxComponent(Component):
    def __init__(self, sim, clk):
        self.stream_slave = StreamIFSlave(sim=sim, bitwidth=32)
        self.schema_slave = SchemaTransferIFSlave(
            sim=sim,
            transport=StreamTransport(slave_ep=self.stream_slave),
            schema_type=SensorDU,
            bitwidth=32,
            rx_proc=self._dispatch,
        )

    _handlers = {
        TempPacket:  lambda self, p: self._on_temp(p),
        PressPacket: lambda self, p: self._on_press(p),
        AccelPacket: lambda self, p: self._on_accel(p),
    }

    def _dispatch(self, du: SensorDU) -> ProcessGen:
        handler = self._handlers.get(type(du.payload))
        if handler:
            yield from handler(self, du.payload)

    def _on_temp(self, p: TempPacket) -> ProcessGen:
        print(f"Temp: {p.temp_raw}  sensor={p.sensor_id}")
        yield self.timeout(0)

    def _on_accel(self, p: AccelPacket) -> ProcessGen:
        print(f"Accel: ax={p.ax}  ay={p.ay}  az={p.az}")
        yield self.timeout(0)

    def _on_press(self, p: PressPacket) -> ProcessGen:
        print(f"Pressure: {p.pressure_pa}  sensor={p.sensor_id}")
        yield self.timeout(0)
```

Wire footprint: `SensorDU.nwords_per_inst(32) = 3` words per transfer
(1 header + 2 payload words, padded to `AccelPacket`'s 48 bits).

---

## 9. Dispatch Patterns on the Receive Side

In single-type mode `rx_proc` receives the deserialized `DataSchema` directly.

In multi-type mode `rx_proc` receives a `DataUnion` instance. The caller accesses
`.payload` and `type(.payload)` for dispatch:

```python
# dispatch table (recommended)
_handlers: dict[type, Callable] = {
    TempPacket:  _on_temp,
    AccelPacket: _on_accel,
}

def _dispatch(self, du: SensorDU) -> ProcessGen:
    handler = self._handlers.get(type(du.payload))
    if handler:
        yield from handler(self, du.payload)

# poll-style — block until next object arrives
async def run_proc(self):
    while True:
        event = self.schema_slave.queue.get()
        yield event
        du = event.value           # SensorDU in multi-type mode
        # du.payload, type(du.payload), du.schema_id all available
```

---

## 10. Open Questions

1. **`bitwidth` inference**: Should `SchemaTransferIF.bind()` infer `bitwidth` from
   the physical transport, or always require it to be set explicitly?
2. **AXI-MM address management**: `AXIMMTransport.base_addr` must not conflict with
   other AXI-MM slaves. Should `SchemaTransferIF` register with `assign_address_ranges`,
   or leave address assignment to the system integrator?
3. **Unknown `schema_id` handling**: In multi-type mode, `SchemaIDField._convert()`
   already raises `ValueError` for unregistered IDs during `DataUnion.deserialize()`.
   Should the slave catch this and route to an optional `rx_unknown_proc` callback,
   or let it propagate?
4. **Read-back / request-response**: Should `SchemaTransferIF` support reading a schema
   back through `AXIMMTransport`, enabling a request/response protocol?
5. **Transport lifetime**: Should `SchemaTransferIF` take ownership of
   `PhysicalTransport` and call `pre_sim` on it, or leave lifecycle to the `Component`?

---

## 11. Proposed File Layout

```
pysilicon/hw/
    dataunion.py                    # already complete:
                                    #   SchemaRegistry, register_schema,
                                    #   SchemaIDField, LengthField,
                                    #   DataUnionHdr, DataUnion

    schema_transfer_interface.py    # to be implemented:
                                    #   PhysicalTransport (ABC),
                                    #   StreamTransport, AXIMMTransport, CrossBarTransport,
                                    #   SchemaTransferIFMaster, SchemaTransferIFSlave,
                                    #   SchemaTransferIF

tests/hw/
    test_dataunion.py               # already complete (144 tests)
    test_dataunion_vitis.py         # already complete (vitis loopback)
    test_schema_transfer_interface.py  # to be implemented:
                                       #   single-type round-trip
                                       #   multi-type round-trip via DataUnion
                                       #   StreamTransport integration (SimPy)
                                       #   AXIMMTransport integration

examples/interface/
    schema_transfer_demo.py         # end-to-end demo, both modes
```

---

## 12. Implementation Sequence

1. ~~**`SchemaRegistry` + `register_schema`**~~ — complete (`dataunion.py`)
2. ~~**`DataUnion` + `DataUnionHdr`**~~ — complete (`dataunion.py`)
3. **`PhysicalTransport` ABC + `StreamTransport`**: simplest physical layer; enables
   most of the logical interface to be built and tested without AXI-MM.
4. **`SchemaTransferIFMaster`, `SchemaTransferIFSlave`, `SchemaTransferIF`**: core
   logical layer; write unit tests using `StreamTransport` in both single-type and
   multi-type modes.
5. **`AXIMMTransport` + `CrossBarTransport`**: remaining physical layer adapters.
6. **Integration example** (`schema_transfer_demo.py`): end-to-end demo showing both
   modes with `StreamIF` as the physical layer.
