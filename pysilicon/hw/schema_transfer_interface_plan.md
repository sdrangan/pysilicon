# SchemaTransferIF — Design Plan

## 1. Overview

`SchemaTransferIF` is a logical interface for transmitting and receiving `DataSchema` objects
over physical transport mechanisms such as `StreamIF` and `AXIMMCrossBarIF`. It leverages
`DataSchema`'s existing `serialize()` / `deserialize()` methods to convert structured Python
objects to and from raw word arrays, and adds a compact wire header so the receiver can
reconstruct the correct schema type without out-of-band signaling.

The design follows the same layering pattern already present in the codebase:

```
Application layer:  Component.write_schema(schema)  /  rx_schema_proc(schema, schema_cls)
                        |                                        |
Logical layer:   SchemaTransferIFMaster          SchemaTransferIFSlave
                        |                                        |
Transport layer: PhysicalTransport  (StreamTransport | AXIMMTransport | CrossBarTransport)
                        |                                        |
Physical layer:  StreamIFMaster / AXIMMCrossBarIFMaster   StreamIFSlave / AXIMMCrossBarIFSlave
```

---

## 2. Goals

1. Allow a `Component` to send any registered `DataSchema` instance through a single
   `write_schema(schema)` call, regardless of the underlying physical interface.
2. Allow a `Component` to receive `DataSchema` instances via a slave endpoint callback
   `rx_schema_proc(schema_instance, schema_cls)` that provides both the deserialized object
   and its Python class.
3. Support multiple `DataSchema` types on the same interface using a schema type ID embedded
   in the wire header.
4. Decouple the logical interface from the physical layer through a `PhysicalTransport`
   abstraction, so a single `SchemaTransferIF` can back any physical transport.

---

## 3. Non-Goals / Out of Scope (Initial Version)

- Flow control or acknowledgment beyond what the underlying physical layer already provides.
- Schema versioning or migration.
- HLS code generation for `SchemaTransferIF` endpoints (deferred — requires extending
  `DataSchema.gen_write` / `gen_read`).
- Streaming fragmentation for schemas larger than the physical layer's maximum burst size.

---

## 4. Wire Protocol

Each transmission is a single burst on the physical interface. The burst consists of:

```
Word 0:   [31:16] schema_id   [15:0] num_payload_words
Words 1…N: serialize(schema, word_bw=bitwidth)     (N = num_payload_words)
```

- **`schema_id`** (16 bits): integer uniquely identifying the `DataSchema` subclass;
  stored in the `SchemaRegistry`.
- **`num_payload_words`** (16 bits): number of payload words; allows the slave to consume
  exactly the right slice of the burst even if the burst carries trailing padding.
- The header is always one word, regardless of the configured `bitwidth`. For 64-bit
  transports the upper half of Word 0 is reserved as zero.

This format is transport-agnostic and works for both stream and memory-mapped transports.
If `num_payload_words` needs to exceed 65 535 (i.e., for very large schemas), the design
can be extended to a two-word header in a later revision.

---

## 5. Schema Registry

A `SchemaRegistry` maintains a bidirectional mapping between integer IDs and
`DataSchema` subclasses. It is intentionally separate from `DataSchema` itself so that
a single class can participate in multiple registries with different IDs (e.g., in
multi-subsystem designs).

```python
class SchemaRegistry:
    def register(self, schema_cls: type[DataSchema], schema_id: int) -> None: ...
    def get_id(self, schema_cls: type[DataSchema]) -> int: ...
    def get_class(self, schema_id: int) -> type[DataSchema]: ...
```

A module-level singleton `DEFAULT_SCHEMA_REGISTRY` is provided so simple designs do not
need to manage registry lifetime. A class decorator `@register_schema` makes registration
ergonomic:

```python
DEFAULT_SCHEMA_REGISTRY = SchemaRegistry()

def register_schema(schema_id: int, registry: SchemaRegistry = DEFAULT_SCHEMA_REGISTRY):
    """Class decorator: registers a DataSchema subclass in the given registry."""
    def decorator(cls: type[DataSchema]) -> type[DataSchema]:
        registry.register(cls, schema_id)
        return cls
    return decorator
```

Usage:

```python
@register_schema(schema_id=1)
class SensorPacket(DataList):
    elements = {"x": Int32, "y": Int32}

@register_schema(schema_id=2)
class ControlWord(DataList):
    elements = {"opcode": UInt8, "payload": UInt32}
```

---

## 6. Physical Transport Abstraction

`PhysicalTransport` is an abstract base class with two responsibilities:
(a) sending a word burst, and (b) registering a receive callback.

```python
class PhysicalTransport(ABC):

    @abstractmethod
    def write_words(self, words: Words) -> ProcessGen:
        """Transmit a word burst through the physical endpoint."""
        ...

    @abstractmethod
    def set_rx_callback(self, callback: Callable[[Words], ProcessGen]) -> None:
        """Register the callback to be invoked when a word burst arrives."""
        ...
```

### 6.1 StreamTransport

Wraps a `StreamIFMaster` (TX path) or a `StreamIFSlave` (RX path).

```python
class StreamTransport(PhysicalTransport):
    master_ep: StreamIFMaster | None = None
    slave_ep: StreamIFSlave | None = None

    def write_words(self, words: Words) -> ProcessGen:
        # delegates to StreamIFMaster.write(words)
        yield from self.master_ep.write(words)

    def set_rx_callback(self, callback: Callable[[Words], ProcessGen]) -> None:
        # sets StreamIFSlave.rx_proc to the callback
        self.slave_ep.rx_proc = callback
```

`StreamTransport` is constructed with either a master endpoint (for TX-side
`SchemaTransferIFMaster`) or a slave endpoint (for RX-side `SchemaTransferIFSlave`).

### 6.2 AXIMMTransport

Wraps `AXIMMCrossBarIFMaster` (TX) or `AXIMMCrossBarIFSlave` (RX). Uses a single
dedicated base address for schema burst writes; the address is agreed upon at system
configuration time.

```python
class AXIMMTransport(PhysicalTransport):
    master_ep: AXIMMCrossBarIFMaster | None = None
    slave_ep: AXIMMCrossBarIFSlave | None = None
    base_addr: int = 0x0000

    def write_words(self, words: Words) -> ProcessGen:
        yield from self.master_ep.write(words, self.base_addr)

    def set_rx_callback(self, callback: Callable[[Words], ProcessGen]) -> None:
        # wraps callback as AXIMMCrossBarIFSlave.rx_write_proc(words, local_addr)
        self.slave_ep.rx_write_proc = lambda words, addr: callback(words)
```

`base_addr` must be within the `AXIMMAddressRange` assigned to the slave endpoint via the
existing `assign_address_ranges` utility.

### 6.3 CrossBarTransport

Wraps `CrossBarIFInput` (TX) or `CrossBarIFOutput` (RX), using the same delegation
pattern as `StreamTransport`.

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

## 7. SchemaTransferIF Endpoints and Interface

### 7.1 SchemaTransferIFMaster

TX-side endpoint owned by a `Component`. The `Component` also owns the underlying
physical endpoint (e.g., `StreamIFMaster`) and passes it to `StreamTransport` at
construction time.

```python
class SchemaTransferIFMaster(InterfaceEndpoint):
    transport: PhysicalTransport
    registry: SchemaRegistry = DEFAULT_SCHEMA_REGISTRY
    bitwidth: int = 32

    def write_schema(self, schema: DataSchema) -> ProcessGen:
        schema_id = self.registry.get_id(type(schema))
        payload = schema.serialize(word_bw=self.bitwidth)   # returns np.ndarray
        num_words = len(payload)
        header = np.array([(schema_id << 16) | num_words], dtype=np.uint32)
        burst = np.concatenate([header, payload])
        yield from self.transport.write_words(burst)
```

### 7.2 SchemaTransferIFSlave

RX-side endpoint. The `Component` provides an `rx_schema_proc` callback to receive
deserialized schemas.

```python
class SchemaTransferIFSlave(InterfaceEndpoint):
    transport: PhysicalTransport
    registry: SchemaRegistry = DEFAULT_SCHEMA_REGISTRY
    bitwidth: int = 32
    rx_schema_proc: Callable[[DataSchema, type[DataSchema]], ProcessGen] | None = None
    schema_queue: simpy.Store   # populated in pre_sim(); enables poll-style RX

    def pre_sim(self) -> None:
        self.schema_queue = simpy.Store(self.sim.env)
        self.transport.set_rx_callback(self._on_words_received)

    def _on_words_received(self, words: Words) -> ProcessGen:
        header = int(words[0])
        schema_id = (header >> 16) & 0xFFFF
        num_payload = header & 0xFFFF
        payload = words[1: 1 + num_payload]
        schema_cls = self.registry.get_class(schema_id)
        schema_inst = schema_cls()
        schema_inst.deserialize(payload, word_bw=self.bitwidth)
        yield self.schema_queue.put((schema_inst, schema_cls))
        if self.rx_schema_proc is not None:
            yield from self.rx_schema_proc(schema_inst, schema_cls)
```

`schema_queue` provides a `simpy.Store` interface for coroutines that prefer to `yield`
on an incoming schema rather than provide a callback.

### 7.3 SchemaTransferIF

The logical interface acts as a named container for the master/slave endpoint pair.
It does not subclass `QueuedTransferIF` because physical framing is handled by the
transport layer; it only needs to validate that both endpoints share the same registry
and bitwidth.

```python
class SchemaTransferIF(Interface):
    endpoint_names = ('master', 'slave')
    registry: SchemaRegistry = DEFAULT_SCHEMA_REGISTRY

    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        # Validate endpoint is SchemaTransferIFMaster or SchemaTransferIFSlave
        # Validate registry and bitwidth consistency
        super().bind(ep_name, endpoint)
```

---

## 8. Associating Physical Endpoints with SchemaTransferIF

Each `Component` that participates in schema transfer owns both a physical endpoint
(e.g., `StreamIFMaster`) and a `SchemaTransferIFMaster`. The physical endpoint is passed
to the transport at construction time, making the association explicit:

```python
class TxComponent(Component):
    def __init__(self, sim, ...):
        # Physical endpoint — registered with a StreamIF in the usual way
        self.stream_master = StreamIFMaster(sim=sim, bitwidth=32)
        self.add_endpoint(self.stream_master)

        # Logical endpoint — wraps the physical endpoint via StreamTransport
        self.schema_master = SchemaTransferIFMaster(
            sim=sim,
            transport=StreamTransport(master_ep=self.stream_master),
            registry=my_registry,
            bitwidth=32,
        )
        self.add_endpoint(self.schema_master)
```

```python
class RxComponent(Component):
    def __init__(self, sim, ...):
        self.stream_slave = StreamIFSlave(sim=sim, bitwidth=32)
        self.add_endpoint(self.stream_slave)

        self.schema_slave = SchemaTransferIFSlave(
            sim=sim,
            transport=StreamTransport(slave_ep=self.stream_slave),
            registry=my_registry,
            bitwidth=32,
            rx_schema_proc=self._handle_schema,
        )
        self.add_endpoint(self.schema_slave)

    async def _handle_schema(
        self, schema: DataSchema, schema_cls: type[DataSchema]
    ) -> ProcessGen:
        if isinstance(schema, SensorPacket):
            yield from self._process_sensor(schema)
        elif isinstance(schema, ControlWord):
            yield from self._process_control(schema)
```

Connecting at the system level:

```python
stream_if = StreamIF(sim=sim, clk=clk, bitwidth=32)
stream_if.bind("master", tx_comp.stream_master)
stream_if.bind("slave", rx_comp.stream_slave)

schema_if = SchemaTransferIF(sim=sim, registry=my_registry)
schema_if.bind("master", tx_comp.schema_master)
schema_if.bind("slave", rx_comp.schema_slave)
```

Both the `StreamIF` and the `SchemaTransferIF` are declared; the `SchemaTransferIF` does
not replace the physical interface but rather layers on top of it.

---

## 9. Multi-Schema Dispatch on the RX Side

The `rx_schema_proc` callback receives `(schema_instance, schema_cls)`. The recommended
dispatch pattern is a simple `isinstance` chain (readable, type-checked by mypy) or a
dictionary keyed by `schema_cls`:

```python
# isinstance-based (recommended for small number of types)
async def _handle_schema(self, schema, schema_cls):
    if isinstance(schema, SensorPacket):
        ...
    elif isinstance(schema, ControlWord):
        ...

# dispatch-table (preferred for large or extensible type sets)
_handlers = {
    SensorPacket: _handle_sensor,
    ControlWord:  _handle_control,
}

async def _handle_schema(self, schema, schema_cls):
    handler = self._handlers.get(schema_cls)
    if handler:
        yield from handler(self, schema)
```

For poll-style designs (e.g., a coroutine that blocks until a specific schema type arrives):

```python
async def run_proc(self):
    while True:
        event = self.schema_slave.schema_queue.get()
        yield event
        schema, schema_cls = event.value
        # process schema
```

---

## 10. Open Questions

1. **Header word width for large schemas**: 16 bits for `num_payload_words` supports at
   most 65 535 words × 4 bytes = 256 KB per transfer. For larger schemas, a two-word
   header (or a 32-bit length field) would be needed.
2. **Bitwidth negotiation**: Should `SchemaTransferIF.bind()` infer `bitwidth` from the
   physical transport (mirroring how `QueuedTransferIF._validate_and_set_bitwidth` works),
   or should it always be set explicitly?
3. **AXI-MM address management**: `AXIMMTransport.base_addr` must not conflict with other
   AXI-MM slaves. Should `SchemaTransferIF` register itself with `assign_address_ranges`,
   or leave address assignment entirely to the system integrator?
4. **Read-back / request-response**: Should `SchemaTransferIF` support reading a schema
   back through `AXIMMTransport` (using `rx_read_proc`)? This would enable a
   request/response protocol where a master writes a request schema and reads a response
   schema from the same address range.
5. **Unknown schema_id handling**: If the slave receives an unregistered `schema_id`,
   should it silently discard the burst, log a warning, raise an exception, or call an
   optional `rx_unknown_proc` callback?
6. **Transport lifetime and ownership**: The `PhysicalTransport` objects hold references to
   physical endpoints. Should `SchemaTransferIF` take ownership (and call `pre_sim` on
   them), or should the owning `Component` manage transport lifecycle?

---

## 11. Proposed File Layout

```
pysilicon/hw/
    schema_transfer_interface.py   # SchemaRegistry, register_schema decorator,
                                   # PhysicalTransport (ABC),
                                   # StreamTransport, AXIMMTransport, CrossBarTransport,
                                   # SchemaTransferIFMaster, SchemaTransferIFSlave,
                                   # SchemaTransferIF

tests/hw/
    test_schema_transfer_interface.py   # unit tests:
                                        #   - serialize/deserialize round-trip
                                        #   - header encoding/decoding
                                        #   - multi-schema dispatch
                                        #   - StreamTransport integration
                                        #   - AXIMMTransport integration

examples/interface/
    schema_transfer_demo.py        # end-to-end demo using StreamIF as physical layer
```

---

## 12. Implementation Sequence

1. **DataSchema extension** (optional, minimal): Confirm that `serialize()` /
   `deserialize()` are sufficient as-is. No changes to `DataSchema` are required for the
   core design; `schema_id` lives in `SchemaRegistry`, not on the class.
2. **`SchemaRegistry` + `register_schema` decorator**: Implement and test in isolation.
3. **`PhysicalTransport` ABC + `StreamTransport`**: The simplest physical layer; enables
   most of the logical interface to be built and tested without AXI-MM.
4. **`SchemaTransferIFMaster`, `SchemaTransferIFSlave`, `SchemaTransferIF`**: Core logical
   layer; write unit tests using `StreamTransport`.
5. **`AXIMMTransport` + `CrossBarTransport`**: Add remaining physical layer adapters with
   corresponding tests.
6. **Integration example** (`schema_transfer_demo.py`): End-to-end demo wiring
   `TxComponent` → `StreamIF` → `RxComponent` with multi-schema dispatch.
