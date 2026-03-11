from abc import ABC, abstractmethod
import json
import math
import re
from pathlib import Path
from enum import IntEnum
from typing import Any, List, Optional, Literal, Tuple, Type
import numpy as np
from numpy.typing import NDArray

class DataSchema(ABC):  
    """
    Base class for HLS Data Structure generation.

    All hardware-mappable types (Scalar, Array, Struct, Union) inherit 
    from this. 

    Parameters
    ----------
    name : str
        The name of this data node, used for C++ field naming and code generation.
    description : Optional[str]
        An optional human-readable description of this field, useful for documentation and code comments.
    """
    def __init__(
            self, 
            name: str, 
            description: Optional[str] = None):
        self.name = name
        self.description = description
        self._val = None  # Holds the Python representation

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, new_val):
        self._val = self._convert(new_val)

    def _convert(self, new_val):
        """
        Attempts to convert new_val to the appropriate Python type for this field.
        If this is not possible, it raises an Exception
        """
        return new_val

    def to_dict(self) -> dict:
        """
        Converts the current value of this DataSchema node into a JSON-serializable dictionary.
        The function simply returns a dictionary with a single key-value pair of `{name: val}`.

        DataList nodes will naturally create a nested dictionary structure.
        """
        return {self.name : self.val}

    def from_dict(self, data: dict) -> "DataSchema":
        """
        Populates this DataSchema node from a dictionary representation.

        The input can be either:
        - a wrapped dictionary from :meth:`to_dict`, e.g. ``{"pkt": {...}}``
        - a direct payload for this node, e.g. ``{"field": 1, ...}``
        """
        if not isinstance(data, dict):
            raise TypeError("from_dict expects a dictionary input.")

        if self.name in data:
            payload = data[self.name]
        else:
            payload = data

        self.val = payload
        return self

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """Convert nested values to types supported by json.dumps."""
        if isinstance(value, dict):
            return {k: DataSchema._to_jsonable(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [DataSchema._to_jsonable(v) for v in value]

        if isinstance(value, np.ndarray):
            return [DataSchema._to_jsonable(v) for v in value.tolist()]

        if isinstance(value, np.generic):
            return value.item()

        return value

    def to_json(
        self,
        file_path: Optional[str | Path] = None,
        indent: Optional[int] = 2,
    ) -> str:
        """
        Serializes :meth:`to_dict` output to a JSON string.

        If ``file_path`` is provided, the JSON content is also written to disk.
        """
        payload = self._to_jsonable(self.to_dict())
        json_str = json.dumps(payload, indent=indent)

        if file_path is not None:
            out_path = Path(file_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_str, encoding="utf-8")

        return json_str

    def from_json(self, json_input: str | Path) -> "DataSchema":
        """
        Loads this DataSchema from JSON.

        ``json_input`` can be either a JSON string or a path to a JSON file.
        """
        if isinstance(json_input, Path):
            raw = json_input.read_text(encoding="utf-8")
        elif isinstance(json_input, str):
            path_candidate = Path(json_input)
            if path_candidate.exists() and path_candidate.is_file():
                raw = path_candidate.read_text(encoding="utf-8")
            else:
                raw = json_input
        else:
            raise TypeError("from_json expects a JSON string or a pathlib.Path.")

        return self.from_dict(json.loads(raw))

    @abstractmethod
    def get_bitwidth(self) -> int:
        """Returns the total number of bits this node occupies in hardware."""
        pass

    @classmethod
    def cpp_class_name(cls) -> str:
        """Gets the C++ class name for this DataSchema type, used in code generation.
        By default, this is the same as the Python class name (e.g., 'ComplexSample').
        For fields, this will be overridden to return the cpp_type (e.g., 'ap_int<16>').
        """
        return cls.__name__

    @abstractmethod
    def gen_pack(self, 
                 indent_level: int = 0) -> str:
        """
        Generates the Vitis C++ code for a 'pack_to_uint' method.

        constexpr int C::bitwidth = ...;
        static ap_uint<C::bitwidth> C::pack_to_uint(const C& data) {
        }

        This function maps the internal fields to a single wide ap_uint<W>.
        """
        raise NotImplementedError("gen_pack must be implemented by subclasses of DataSchema.")

    @abstractmethod
    def gen_unpack(self,
                   indent_level: int = 0) -> str:
        """
        Generates the Vitis C++ code for an 'unpack_from_uint' method.

        C C::unpack_from_uint(const ap_uint<C::bitwidth>& packed) {
        }

        This function maps a packed ap_uint<W> back into the internal fields.
        """
        raise NotImplementedError("gen_unpack must be implemented by subclasses of DataSchema.")

    @abstractmethod
    def default_value(self) -> Any:
        """
        Returns the default Python-side value for this node.
        """
        raise NotImplementedError("default_value must be implemented by subclasses of DataSchema.")
    
    def nwords_per_inst(self, word_bw: int) -> int:
        """
        Utility to compute how many words of a given bandwidth 
        are needed to represent this node.  The computation assuems the same
        packing as in gen_write()
        """
        if word_bw <= 0:
            raise ValueError("word_bw must be positive.")

        final_ipos, final_iword = self._nwords_per_inst_recursive(
            word_bw=word_bw,
            ipos0=0,
            iword0=0,
        )
        return final_iword + (1 if final_ipos > 0 else 0)
    
    def gen_write(
        self,
        word_bw: Optional[int] = None,
        dst_type: Literal["array", "stream", "axi4_stream"] = "array",
        params: Any = None,
        word_bw_supported: Optional[List[int]] = None,
        indent_level: int = 0,
    ) -> str:
        """
        Generates the Vitis HLS C++ function for writing the data structure 
        to a specific hardware interface.  The code generated will depend on the dst_type and
        will look like one of the following templates:

        if dst_type == "array":
            template<>
            void C::write_array(ap_uint<word_bw> x[]) { 
                // body generated by _gen_write_recursive, which will contain lines like:
                x[0] = ...;
                x[1] = ...;
            }

        elif dst_type == "stream":
            template<>
            void C::write_stream(hls::stream<ap_uint<word_bw>> &s) {
                // body generated by _gen_write_recursive, which will contain lines like:
                ap_uint<word_bw> w =...;
                s.write(w);
                ap_uint<word_bw> w =...;
                s.write(w);
            }

        elif dst_type == "axi4_stream":
            template<>
            void C::write_axi4_stream(
                hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, bool tlast = true) { 
                 // body generated by _gen_write_recursive, which will contain lines like:
                ap_uint<word_bw> w =...;
                streamutils::write_axi4_word<word_bw>(s, w, tlast);
            }
        
        where `C` is the C++ class name corresponding to this DataSchema and `...`
        is the body generated by the recursive engine `_gen_write_recursive`.

        Parameters
        ----------
        word_bw : int
            The bitwidth of the hardware interface word. This determines how the data 
            fields are packed into words.
        dst_type : Literal["array", "stream", "axi4_stream"]
            The type of hardware interface to generate code for. 
            This affects the function signature and how data is written.
        params : Any
            Optional parameters that can be used to control code generation.
            For example, for dynamic sized matrices, the paramters could include a tuple
            max_shape = (max_rows, max_cols) that defines the maximum dimensions 
            for the generated code.  Then, the write function can add the additional parameters
            to the function signature and use them in the body to handle dynamic sizes.
        """
        if word_bw_supported is None:
            if word_bw is None:
                raise ValueError("Either word_bw or word_bw_supported must be provided.")
            word_bw_supported = [word_bw]

        if not word_bw_supported:
            raise ValueError("word_bw_supported must contain at least one value.")

        for bw in word_bw_supported:
            if bw <= 0:
                raise ValueError(f"word_bw values must be positive. Got {bw}.")

        indent = self._get_indent(indent_level)
        i1 = self._get_indent(indent_level + 1)
        i2 = self._get_indent(indent_level + 2)
        i3 = self._get_indent(indent_level + 3)

        param_str = self.get_param_str(params, write=True)

        if dst_type == "array":
            signature = f"{indent}void write_array(ap_uint<word_bw> x[]"
            if param_str:
                signature += f", {param_str}"
            signature += ") const {"
            target = "x"
            unsupported_msg = "Unsupported word_bw for write_array"
        elif dst_type == "stream":
            signature = f"{indent}void write_stream(hls::stream<ap_uint<word_bw>> &s"
            if param_str:
                signature += f", {param_str}"
            signature += ") const {"
            target = "s"
            unsupported_msg = "Unsupported word_bw for write_stream"
        else:
            signature = (
                f"{indent}void write_axi4_stream("
                f"hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, bool tlast = true"
            )
            if param_str:
                signature += f", {param_str}"
            signature += ") const {"
            target = "s"
            unsupported_msg = "Unsupported word_bw for write_axi4_stream"

        lines: List[str] = [
            f"{indent}template<int word_bw>",
            signature,
        ]

        for idx, bw in enumerate(word_bw_supported):
            cond = "if constexpr" if idx == 0 else "else if constexpr"
            lines.append(f"{i1}{cond} (word_bw == {bw}) {{")

            if dst_type != "array":
                lines.append(f"{i2}ap_uint<{bw}> w = 0;")

            final_lines, final_ipos, _ = self._gen_write_recursive(
                word_bw=bw,
                dst_type=dst_type,
                target=target,
                ipos0=0,
                iword0=0,
                prefix="this->",
                params=params,
            )

            if dst_type == "axi4_stream" and final_ipos == 0:
                marker = f"streamutils::write_axi4_word<{bw}>({target}, w, "
                for i in range(len(final_lines) - 1, -1, -1):
                    if marker in final_lines[i]:
                        final_lines[i] = final_lines[i].replace(", false);", ", tlast);")
                        break

            for line in final_lines:
                if line.startswith("    "):
                    line = line[4:]
                lines.append(f"{i2}{line}" if line else "")

            if dst_type != "array" and final_ipos > 0:
                if dst_type == "stream":
                    lines.append(f"{i2}{target}.write(w);")
                else:
                    lines.append(f"{i2}streamutils::write_axi4_word<{bw}>({target}, w, tlast);")

            lines.append(f"{i1}}}")

        lines.extend([
            f"{i1}else {{",
            f"{i2}static_assert(word_bw > 0, \"{unsupported_msg}\");",
            f"{i1}}}",
            f"{indent}}}",
        ])

        return "\n".join(lines)

    def gen_read(
        self,
        word_bw: Optional[int] = None,
        src_type: Literal["array", "stream", "axi4_stream"] = "array",
        params: Any = None,
        word_bw_supported: Optional[List[int]] = None,
        indent_level: int = 0,
    ) -> str:
        """
        Generates the Vitis HLS C++ function for reading the data structure
        from a specific hardware interface. This is the inverse of ``gen_write``.

        Parameters
        ----------
        word_bw : int
            Bitwidth of the source interface word.
        src_type : Literal["array", "stream", "axi4_stream"]
            Source interface type.
        params : Any
            Optional parameter metadata for schema-specific generation.
        word_bw_supported : Optional[List[int]]
            Set of supported compile-time word widths.
        indent_level : int
            Base indentation level for generated code.
        """
        if word_bw_supported is None:
            if word_bw is None:
                raise ValueError("Either word_bw or word_bw_supported must be provided.")
            word_bw_supported = [word_bw]

        if not word_bw_supported:
            raise ValueError("word_bw_supported must contain at least one value.")

        for bw in word_bw_supported:
            if bw <= 0:
                raise ValueError(f"word_bw values must be positive. Got {bw}.")

        indent = self._get_indent(indent_level)
        i1 = self._get_indent(indent_level + 1)
        i2 = self._get_indent(indent_level + 2)

        param_str = self.get_param_str(params, write=False)

        if src_type == "array":
            signature = f"{indent}void read_array(const ap_uint<word_bw> x[]"
            if param_str:
                signature += f", {param_str}"
            signature += ") {"
            source = "x"
            unsupported_msg = "Unsupported word_bw for read_array"
        elif src_type == "stream":
            signature = f"{indent}void read_stream(hls::stream<ap_uint<word_bw>> &s"
            if param_str:
                signature += f", {param_str}"
            signature += ") {"
            source = "s"
            unsupported_msg = "Unsupported word_bw for read_stream"
        else:
            signature = (
                f"{indent}void read_axi4_stream("
                f"hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s"
            )
            if param_str:
                signature += f", {param_str}"
            signature += ") {"
            source = "s"
            unsupported_msg = "Unsupported word_bw for read_axi4_stream"

        lines: List[str] = [
            f"{indent}template<int word_bw>",
            signature,
        ]

        for idx, bw in enumerate(word_bw_supported):
            cond = "if constexpr" if idx == 0 else "else if constexpr"
            lines.append(f"{i1}{cond} (word_bw == {bw}) {{")

            if src_type == "stream":
                lines.append(f"{i2}ap_uint<{bw}> w = 0;")
            elif src_type == "axi4_stream":
                lines.append(f"{i2}ap_uint<{bw}> w = 0;")

            final_lines, _, _ = self._gen_read_recursive(
                word_bw=bw,
                src_type=src_type,
                source=source,
                ipos0=0,
                iword0=0,
                prefix="this->",
                params=params,
            )

            for line in final_lines:
                if line.startswith("    "):
                    line = line[4:]
                lines.append(f"{i2}{line}" if line else "")

            lines.append(f"{i1}}}")

        lines.extend([
            f"{i1}else {{",
            f"{i2}static_assert(word_bw > 0, \"{unsupported_msg}\");",
            f"{i1}}}",
            f"{indent}}}",
        ])

        return "\n".join(lines)

    def gen_dump_json(self, indent_level: int = 0) -> str:
        """Generate C++ code that writes this schema instance as JSON to an output stream."""
        indent = self._get_indent(indent_level)
        i1 = self._get_indent(indent_level + 1)

        lines: List[str] = [
            f"{indent}void dump_json(std::ostream& os, int indent = 2) const {{",
            f"{i1}const int step = (indent < 0) ? 0 : indent;",
        ]

        body = self._gen_dump_json_recursive(
            prefix="this->",
            os_name="os",
            depth_expr="0",
            indent_var="step",
        )
        for line in body:
            if line.startswith("    "):
                line = line[4:]
            lines.append(f"{i1}{line}" if line else "")

        lines.append(f"{indent}}}")
        return "\n".join(lines)

    def gen_load_json(self, indent_level: int = 0) -> str:
        """Generate C++ code that reads this schema instance from a JSON input stream."""
        indent = self._get_indent(indent_level)
        i1 = self._get_indent(indent_level + 1)

        lines: List[str] = [
            f"{indent}void load_json(std::istream& is) {{",
            f"{i1}std::string json_text((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());",
            f"{i1}size_t pos = 0;",
            f"{i1}_json_skip_ws(json_text, pos);",
        ]

        body = self._gen_load_json_recursive(
            prefix="this->",
            json_var="json_text",
            pos_var="pos",
            ctx="root",
        )
        for line in body:
            if line.startswith("    "):
                line = line[4:]
            lines.append(f"{i1}{line}" if line else "")

        lines.extend([
            f"{i1}_json_skip_ws(json_text, pos);",
            f"{i1}if (pos != json_text.size()) {{",
            f"{i1}    throw std::runtime_error(\"Trailing characters after JSON object.\");",
            f"{i1}}}",
            f"{indent}}}",
        ])
        return "\n".join(lines)

    def gen_json_helpers(self, indent_level: int = 0) -> str:
        """Generate small JSON tokenizer/parser helpers used by generated load_json."""
        indent = self._get_indent(indent_level)
        i1 = self._get_indent(indent_level + 1)
        i2 = self._get_indent(indent_level + 2)

        lines: List[str] = [
            f"{indent}static void _json_skip_ws(const std::string& s, size_t& pos) {{",
            f"{i1}while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {{",
            f"{i2}++pos;",
            f"{i1}}}",
            f"{indent}}}",
            "",
            f"{indent}static void _json_expect_char(const std::string& s, size_t& pos, char ch) {{",
            f"{i1}_json_skip_ws(s, pos);",
            f"{i1}if (pos >= s.size() || s[pos] != ch) {{",
            f"{i2}throw std::runtime_error(\"Malformed JSON: unexpected delimiter.\");",
            f"{i1}}}",
            f"{i1}++pos;",
            f"{indent}}}",
            "",
            f"{indent}static std::string _json_parse_string(const std::string& s, size_t& pos) {{",
            f"{i1}_json_skip_ws(s, pos);",
            f"{i1}if (pos >= s.size() || s[pos] != '\"') {{",
            f"{i2}throw std::runtime_error(\"Malformed JSON: expected string key.\");",
            f"{i1}}}",
            f"{i1}++pos;",
            f"{i1}std::string out;",
            f"{i1}while (pos < s.size()) {{",
            f"{i2}char c = s[pos++];",
            f"{i2}if (c == '\\\"') {{",
            f"{i2}    return out;",
            f"{i2}}}",
            f"{i2}if (c == '\\\\') {{",
            f"{i2}    if (pos >= s.size()) {{",
            f"{i2}        throw std::runtime_error(\"Malformed JSON: invalid escape sequence.\");",
            f"{i2}    }}",
            f"{i2}    char esc = s[pos++];",
            f"{i2}    switch (esc) {{",
            f"{i2}    case '\\\"': out.push_back('\\\"'); break;",
            f"{i2}    case '\\\\': out.push_back('\\\\'); break;",
            f"{i2}    case '/': out.push_back('/'); break;",
            f"{i2}    case 'b': out.push_back('\\b'); break;",
            f"{i2}    case 'f': out.push_back('\\f'); break;",
            f"{i2}    case 'n': out.push_back('\\n'); break;",
            f"{i2}    case 'r': out.push_back('\\r'); break;",
            f"{i2}    case 't': out.push_back('\\t'); break;",
            f"{i2}    default:",
            f"{i2}        throw std::runtime_error(\"Malformed JSON: unsupported escape sequence.\");",
            f"{i2}    }}",
            f"{i2}}} else {{",
            f"{i2}    out.push_back(c);",
            f"{i2}}}",
            f"{i1}}}",
            f"{i1}throw std::runtime_error(\"Malformed JSON: unterminated string.\");",
            f"{indent}}}",
            "",
            f"{indent}static double _json_parse_number(const std::string& s, size_t& pos) {{",
            f"{i1}_json_skip_ws(s, pos);",
            f"{i1}if (pos >= s.size()) {{",
            f"{i2}throw std::runtime_error(\"Malformed JSON: expected number.\");",
            f"{i1}}}",
            f"{i1}const char* begin = s.c_str() + pos;",
            f"{i1}char* end = nullptr;",
            f"{i1}double value = std::strtod(begin, &end);",
            f"{i1}if (end == begin) {{",
            f"{i2}throw std::runtime_error(\"Malformed JSON: invalid numeric value.\");",
            f"{i1}}}",
            f"{i1}pos += static_cast<size_t>(end - begin);",
            f"{i1}return value;",
            f"{indent}}}",
        ]

        return "\n".join(lines)

    def get_param_str(
            self, 
            params: Any,
            write: bool) -> str:
        """
        Add parameter specific string in the read and write methods.
        For example, for dynamic sized matrices, the paramters could include a tuple
        "int n0, int n1" that defines the dimensions of the matrix instance being written.

        Parameters
        ----------
        params : Any
            The parameters passed to gen_write or gen_read that can be used to control code generation.
        write : bool
            Whether this is for the write function (True) or read function (False). This can be used to generate different parameter strings for read vs write if needed.
        """
        return ""

    def gen_write_mult(
        self, 
        word_bw: int, 
        dst_type: Literal["array", "stream", "axi4_stream"] = "array"
    ) -> str:
        """
        Generates the C++ function for writing multiple instances of a data structure 
        to a specific hardware interface.  The code generated will depend on the dst_type and
        will look like one of the following templates:

        if dst_type == "array":
            template<>
            void write_array_mult<C,word_bw>(const C src[], ap_uint<word_bw> dst[], int n) { ...  }

        elif dst_type == "stream":
            template<>
            void write_stream_mult<C,word_bw>(const C src[],hls::stream<ap_uint<word_bw>> &s, int n) {... };

        elif dst_type == "axi4_stream":
            template<>
            void write_axi4_stream_mult<C,word_bw>(
                hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, int n, bool tlast = true) { 
            }
        
        where `C` is the C++ class name corresponding to this DataSchema and `...`
        is the body and `n` is the number of instances to write.  There are two cases,
        depending on the packing factor:

            pf = floor(word_bw / C.bitwidth)

        Case 1:  If pf >= 2, we can pack two or more instances into a each word.

        In this case, for example, for dst_type="array" case, the code will look like:
            for (int i = 0; i < n/pf; i++) {
                # pragma to pipeline this loop
                for (int j = 0; j < pf; j++) {
                    # pragma to unroll this inner loop
                    ap_uint<word_bw> w = 0;
                    w.range(...) = {to_uint_expr}(data[i*pf + j]);
                dst[i] = w;
            // Code to hand the final partial word if n is not divisible by pf
                

        Case 2: If pf < 2, we pack only one instance per word, and the code will look like:
            for (int i = 0; i < n; i++) {
                C::write_array(src[i], &dst[i*words_per_inst]);

        Parameters
        ----------
        word_bw : int
            The bitwidth of the hardware interface word. This determines how the data 
            fields are packed into words.
        dst_type : Literal["array", "stream", "axi4_stream"]
            The type of hardware interface to generate code for. 
            This affects the function signature and how data is written.
        """
        class_name = self.cpp_class_name()
        inst_bw = self.get_bitwidth()
        pf = word_bw // inst_bw if inst_bw > 0 else 0
        words_per_inst = self.nwords_per_inst(word_bw)

        if dst_type == "array":
            header = [
                "template<>",
                f"void write_array_mult<{class_name}, {word_bw}>(const {class_name} src[], ap_uint<{word_bw}> dst[], int n) {{",
            ]
        elif dst_type == "stream":
            header = [
                "template<>",
                f"void write_stream_mult<{class_name}, {word_bw}>(const {class_name} src[], hls::stream<ap_uint<{word_bw}>> &s, int n) {{",
            ]
        else:
            header = [
                "template<>",
                f"void write_axi4_stream_mult<{class_name}, {word_bw}>(const {class_name} src[], hls::stream<hls::axis<ap_uint<{word_bw}>, 0, 0, 0>> &s, int n, bool tlast = true) {{",
            ]

        body_lines = [
            f"    constexpr int inst_bw = {inst_bw};",
            f"    constexpr int pf = {pf};",
            f"    constexpr int words_per_inst = {words_per_inst};",
        ]

        if pf >= 2:
            body_lines.extend([
                "    int out_idx = 0;",
                "    int i = 0;",
            ])

            if dst_type == "axi4_stream":
                body_lines.append("    const int total_words = (n + pf - 1) / pf;")

            body_lines.extend([
                "    for (; i + pf <= n; i += pf) {",
                "        #pragma HLS PIPELINE II=1",
                f"        ap_uint<{word_bw}> w = 0;",
            ])

            for j in range(pf):
                low = j * inst_bw
                high = low + inst_bw - 1
                body_lines.append(
                    f"        w.range({high}, {low}) = {class_name}::pack_to_uint(src[i + {j}]);"
                )

            if dst_type == "array":
                body_lines.append("        dst[out_idx++] = w;")
            elif dst_type == "stream":
                body_lines.append("        s.write(w);")
                body_lines.append("        out_idx++;")
            else:
                body_lines.append("        const bool last = (out_idx == total_words - 1) ? tlast : false;")
                body_lines.append(f"        streamutils::write_axi4_word<{word_bw}>(s, w, last);")
                body_lines.append("        out_idx++;")

            body_lines.append("    }")

            body_lines.extend([
                "    if (i < n) {",
                "        #pragma HLS PIPELINE II=1",
                f"        ap_uint<{word_bw}> w = 0;",
            ])

            for j in range(pf):
                low = j * inst_bw
                high = low + inst_bw - 1
                body_lines.extend([
                    f"        if (i + {j} < n) {{",
                    f"            w.range({high}, {low}) = {class_name}::pack_to_uint(src[i + {j}]);",
                    "        }",
                ])

            if dst_type == "array":
                body_lines.append("        dst[out_idx++] = w;")
            elif dst_type == "stream":
                body_lines.append("        s.write(w);")
            else:
                body_lines.append("        const bool last = (out_idx == total_words - 1) ? tlast : false;")
                body_lines.append(f"        streamutils::write_axi4_word<{word_bw}>(s, w, last);")

            body_lines.append("    }")

        else:
            body_lines.extend([
                "    for (int i = 0; i < n; ++i) {"
            ])

            # Add the pipelining pragma if there is only one word per instance
            if words_per_inst == 1:
                body_lines.append("        #pragma HLS PIPELINE II=1")

            if dst_type == "array":
                if words_per_inst == 1:
                    body_lines.append(f"        dst[i] = {class_name}::pack_to_uint(src[i]);")
                else:
                    body_lines.append(f"        src[i].write_array<{word_bw}>(&dst[i * words_per_inst]);")
            elif dst_type == "stream":
                body_lines.append(f"        src[i].write_stream<{word_bw}>(s);")
            else:
                body_lines.append("        const bool last = (i == n - 1) ? tlast : false;")
                body_lines.append(f"        src[i].write_axi4_stream<{word_bw}>(s, last);")

            body_lines.append("    }")

        return "\n".join(header + body_lines + ["}"])

    @abstractmethod
    def _gen_write_recursive(
        self, 
        word_bw: int, 
        dst_type: Literal["array", "stream", "axi4_stream"] = "array",
        target: Literal["x", "s"] = "x", 
        ipos0: int = 0, 
        iword0: int = 0, 
        prefix: str = "",
        params : Any = None
    ) -> Tuple[List[str], int, int]:
        """
        Recursive helper for the gen_write function.

        Parameters
        ----------
        word_bw : int
            Bitwidth of one output word. This defines the packing boundary.
        dst_type : Literal["array", "stream", "axi4_stream"], optional
            Destination interface kind that controls how completed words are emitted.
        target : Literal["x", "s"], optional
            C++ target variable name used by generated code.
            - ``"x"`` for array output
            - ``"s"`` for stream / axi4_stream output
        ipos0 : int, optional
            Initial bit position within the current output word.
            Must be in ``[0, word_bw)`` when called from valid state.
        iword0 : int, optional
            Initial output word index corresponding to the current packing state.
        prefix : str, optional
            Accessor prefix for nested member expressions in generated C++
            (for example ``"this->"`` or ``"this->header."``).
        params : Any, optional
            See gen_write() for details.

        Returns
        --------
        Returns (list of C++ lines, final_bit_pos, final_word_index).
        Tuple[List[str], int, int]
            ``(lines, final_bit_pos, final_word_index)`` where:
            - ``lines`` is the generated C++ statement list for this subtree,
            - ``final_bit_pos`` is the ending bit offset in the current word,
            - ``final_word_index`` is the ending output word index.

        Notes
        -----
        Implementations should preserve state semantics used by ``gen_write`` so
        that parent/child recursion composes correctly and final flush logic is
        applied only once at the top level.
        """
        raise NotImplementedError("Subclasses must implement _gen_write_recursive")

    @abstractmethod
    def _gen_read_recursive(
        self,
        word_bw: int,
        src_type: Literal["array", "stream", "axi4_stream"] = "array",
        source: Literal["x", "s"] = "x",
        ipos0: int = 0,
        iword0: int = 0,
        prefix: str = "",
        params: Any = None,
    ) -> Tuple[List[str], int, int]:
        """
        Recursive helper for ``gen_read``.

        Returns
        -------
        Tuple[List[str], int, int]
            ``(lines, final_bit_pos, final_word_index)`` for this subtree.
        """
        raise NotImplementedError("Subclasses must implement _gen_read_recursive")

    @abstractmethod
    def _gen_dump_json_recursive(
        self,
        prefix: str,
        os_name: str,
        depth_expr: str,
        indent_var: str,
    ) -> List[str]:
        """Recursive helper for ``gen_dump_json``."""
        raise NotImplementedError("Subclasses must implement _gen_dump_json_recursive")

    @abstractmethod
    def _gen_load_json_recursive(
        self,
        prefix: str,
        json_var: str,
        pos_var: str,
        ctx: str,
    ) -> List[str]:
        """Recursive helper for ``gen_load_json``."""
        raise NotImplementedError("Subclasses must implement _gen_load_json_recursive")

    @abstractmethod
    def _nwords_per_inst_recursive(
        self,
        word_bw: int,
        ipos0: int = 0,
        iword0: int = 0,
    ) -> Tuple[int, int]:
        """
        Recursive helper used by nwords_per_inst.
        Returns (final_bit_pos, final_word_index).
        """
        raise NotImplementedError("Subclasses must implement _nwords_per_inst_recursive")

    def _get_indent(self, level: int) -> str:
        """Utility for consistent C++ code formatting."""
        return "    " * level
    

    def to_uint_expr(
            self,
            prefix : Optional[str] = None) -> str:
        """
        Returns a C++ expression that converts a DataSchema instance
        to an unsigned integer representation.  By default,
        this calls the static pack_to_uint method of the corresponding 
        C++ class.

        For DataField nodes, this will be overridden to return 
        the field value directly (with any necessary conversions).
        """
        if prefix is None:
            prefix = ""
        return f"{self.cpp_class_name()}::pack_to_uint({prefix}{self.name})"

    def from_uint_expr(self, uint_expr: str) -> str:
        """
        Returns a C++ expression that converts an unsigned integer slice back
        to this node's C++ representation.
        """
        return f"{self.cpp_class_name()}::unpack_from_uint({uint_expr})"
    
    def gen_cpp_class(self) -> str:
        """
        Generates the C++ class definition. 

        For example, for DataField, this is just a typedef using cpp_type.
        For DataList, this is a class with fields corresponding to the elements.
        """
        raise NotImplementedError("gen_cpp_class must be implemented by subclasses of DataSchema.")
    

    def serialize(self, word_bw: int = 32) -> NDArray[np.unsignedinteger]:
        """
        Serializes the Python-side value (`self.val`) into a packed bit representation
        compatible with HLS memory-mapped interfaces.

        The packing logic must match the hardware's `gen_write()` implementation,
        aligning `self.val` to the LSB of the hardware word.

        Logic:
        1. If word_bw <= 32:
           Returns a 1D array of `np.uint32`. Each element is one hardware word.
           Padding: If the schema size < 32, the upper bits of the uint32 are zeroed.
        
        2. If 32 < word_bw <= 64:
           Returns a 1D array of `np.uint64`. Each element is one hardware word.
           Padding: If the schema size < 64, the upper bits of the uint64 are zeroed.

        3. If word_bw > 64:
           Returns a 2D array of `np.uint64` with shape (n_words, d), where 
           d = ceil(word_bw / 64). 
           Each row [i, :] represents a single hardware word of `word_bw` bits.
           - The word is split into 64-bit chunks: row[i, 0] contains bits [63:0], 
             row[i, 1] contains bits [127:64], and so on.
           - If `word_bw` is not a multiple of 64, the highest index `d-1` in each 
             row contains the remaining MSBs, with the rest of that uint64 zeroed.

        Parameters
        ----------
        word_bw : int
            The bitwidth of the target HLS interface (e.g., 32, 64, 128, 512).
            Determines the "container" size for each serialized data element.

        Returns
        -------
        NDArray[np.unsignedinteger]
            The serialized data packed into the appropriate NumPy uint format.
        """
        if word_bw <= 0:
            raise ValueError("word_bw must be positive.")

        words: List[int] = [0]
        final_ipos, final_iword = self._serialize_recursive(
            word_bw=word_bw,
            words=words,
            ipos0=0,
            iword0=0,
        )

        n_words = final_iword + (1 if final_ipos > 0 else 0)
        if n_words == 0:
            n_words = 1

        words = words[:n_words]

        if word_bw <= 32:
            return np.array([np.uint32(w & ((1 << min(word_bw, 32)) - 1)) for w in words], dtype=np.uint32)

        if word_bw <= 64:
            return np.array([np.uint64(w & ((1 << word_bw) - 1)) for w in words], dtype=np.uint64)

        chunks_per_word = math.ceil(word_bw / 64)
        out = np.zeros((n_words, chunks_per_word), dtype=np.uint64)
        mask64 = (1 << 64) - 1
        for i, w in enumerate(words):
            for j in range(chunks_per_word):
                out[i, j] = np.uint64((w >> (64 * j)) & mask64)
        return out

    def _serialize_recursive(
        self,
        word_bw: int,
        words: List[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> Tuple[int, int]:
        raise NotImplementedError("Subclasses must implement _serialize_recursive.")

    def deserialize(
        self,
        packed: NDArray[np.unsignedinteger],
        word_bw: int = 32,
    ) -> "DataSchema":
        """
        Deserializes packed hardware-word data into this instance's Python-side values.

        Requirements
        ------------
        - `word_bw` must be positive.
        - The input layout must match the exact output format produced by
          :meth:`serialize` for the same `word_bw`:
            1. `word_bw <= 32`:
               `packed` is a 1D array-like of `uint32` words.
            2. `32 < word_bw <= 64`:
               `packed` is a 1D array-like of `uint64` words.
            3. `word_bw > 64`:
               `packed` is a 2D array-like of shape `(n_words, ceil(word_bw/64))`,
               with chunk order `[63:0], [127:64], ...` per row.
        - Field packing state follows the same recursive boundaries as
          :meth:`serialize` / :meth:`gen_write` (no single field split across words).

        Parameters
        ----------
        packed : NDArray[np.unsignedinteger]
            Packed word data in the format described above.
        word_bw : int, optional
            Bitwidth of each hardware word.

        Returns
        -------
        DataSchema
            Returns `self` after in-place population of field values.
        """
        if word_bw <= 0:
            raise ValueError("word_bw must be positive.")

        arr = np.asarray(packed)
        words: List[int] = []

        if word_bw <= 64:
            if arr.ndim == 0:
                arr = arr.reshape(1)
            elif arr.ndim != 1:
                raise ValueError("For word_bw <= 64, packed must be a 1D array-like.")

            mask = (1 << word_bw) - 1
            words = [int(v) & mask for v in arr]
        else:
            chunks_per_word = math.ceil(word_bw / 64)
            if arr.ndim != 2:
                raise ValueError("For word_bw > 64, packed must be a 2D array-like.")
            if arr.shape[1] != chunks_per_word:
                raise ValueError(
                    f"For word_bw={word_bw}, packed must have shape (n_words, {chunks_per_word})."
                )

            mask = (1 << word_bw) - 1
            for row in arr:
                word = 0
                for j, chunk in enumerate(row):
                    word |= int(np.uint64(chunk)) << (64 * j)
                words.append(word & mask)

        if not words:
            words = [0]

        self._deserialize_recursive(
            word_bw=word_bw,
            words=words,
            ipos0=0,
            iword0=0,
        )
        return self

    def _deserialize_recursive(
        self,
        word_bw: int,
        words: List[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> Tuple[int, int]:
        raise NotImplementedError("Subclasses must implement _deserialize_recursive.")

    def is_close(
            self, 
            other: DataSchema, 
            rel_tol : float | None =  None, 
            abs_tol : float | None = 1e-8) -> bool:
        """
        Compares this schema's values with another. 
        Default implementation uses standard equality.

        Parameters
        ----------
        other : DataSchema
            The other DataSchema instance to compare against.
        rel_tol : float, optional
            Relative tolerance for comparison (used for floating-point fields).
            If None the check is skipped.
        abs_tol : float, optional
            Absolute tolerance for comparison (used for floating-point fields).
            If None the check is skipped.

        Returns
        -------
        bool
            True if the values are considered close/equal, False otherwise.
        """
        if not isinstance(other, self.__class__):
            return False
        return self.val == other.val

    def gen_include(
            self,
            include_dir: Optional[str] = None,
            file_name: Optional[str] = None,
            word_bw_supported : List[int] = None) -> str:
        """
        Generates and writes an include file for the DataSchema.

        The file will have the following structure:

        #ifndef {GUARD_NAME}
        #define {GUARD_NAME}

        // C++ class definition for this DataSchema
        {CPP_CLASS_DEFINITION}

        // Additional helper functions (e.g., pack_to_uint, unpack_from_uint) if needed

        #endif // {GUARD_NAME}

        Parameters
        ----------
        include_dir : Optional[str]
            Directory to place the include file. If None, uses current directory.
        file_name : Optional[str]
            Name of the include file. If None, uses "{self.name}.h" but in snake case.
        word_bw_supported : Optional[List[int]]
            List of word bitwidths to generate write_mult functions for. 
            If None, defaults to [].  For each word_bw in the list, it writes the code
            for C::write_array, C::write_stream, and C::write_axi4_stream_mult functions.

        Returns
        -------
        str
            Path to the written include file.
        """
        if word_bw_supported is None:
            word_bw_supported = []

        for bw in word_bw_supported:
            if bw <= 0:
                raise ValueError(f"word_bw values must be positive. Got {bw}.")

        if file_name is None:
            base_name = self.name if self.name else self.cpp_class_name()
            snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", base_name).lower()
            file_name = f"{snake_name}.h"

        guard_base = file_name.replace(".", "_").replace("-", "_").replace("/", "_").replace("\\", "_")
        guard_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in guard_base).upper()

        lines: List[str] = [
            f"#ifndef {guard_name}",
            f"#define {guard_name}",
            "",
            "#include <ap_int.h>",
            "#include <cctype>",
            "#include <cstdlib>",
            "#include <fstream>",
            "#include <hls_stream.h>",
            "#include <iterator>",
            "#include <stdexcept>",
            "#include <string>",
            "#if __has_include(<hls_axi_stream.h>)",
            "#include <hls_axi_stream.h>",
            "#else",
            "#include <ap_axi_sdata.h>",
            "#endif",
            "#include <iostream>",
            "#include \"streamutils.h\"",
            "",
        ]

        class_name = self.cpp_class_name()
        elements = getattr(self, "elements", None)
        if elements is None:
            raise TypeError("gen_include currently supports only class-like schemas with an 'elements' attribute.")

        lines.extend([
            f"class {class_name} {{",
            "public:",
        ])

        for element in elements:
            lines.append(f"    {element.cpp_class_name()} {element.name};")

        lines.append("")
        lines.append(f"    static constexpr int bitwidth = {self.get_bitwidth()};")
        lines.append("")

        pack_impl = self.gen_pack(indent_level=1)
        if pack_impl:
            lines.append(pack_impl)
            lines.append("")

        unpack_impl = self.gen_unpack(indent_level=1)
        if unpack_impl:
            lines.append(unpack_impl)
            lines.append("")

        for dst_type in ("array", "stream", "axi4_stream"):
            lines.append(
                self.gen_write(
                    dst_type=dst_type,
                    params=None,
                    word_bw_supported=word_bw_supported,
                    indent_level=1,
                )
            )
            lines.append("")

        for src_type in ("array", "stream", "axi4_stream"):
            lines.append(
                self.gen_read(
                    src_type=src_type,
                    params=None,
                    word_bw_supported=word_bw_supported,
                    indent_level=1,
                )
            )
            lines.append("")

        lines.append(self.gen_dump_json(indent_level=1))
        lines.append("")
        lines.append(self.gen_load_json(indent_level=1))
        lines.append("")

        lines.extend([
            "#ifndef __SYNTHESIS__",
            "    void dump_json_file(const char* file_path, int indent = 2) const {",
            "        std::ofstream ofs(file_path);",
            "        if (!ofs) {",
            "            throw std::runtime_error(\"Failed to open output JSON file.\");",
            "        }",
            "        this->dump_json(ofs, indent);",
            "    }",
            "",
            "    void load_json_file(const char* file_path) {",
            "        std::ifstream ifs(file_path);",
            "        if (!ifs) {",
            "            throw std::runtime_error(\"Failed to open input JSON file.\");",
            "        }",
            "        this->load_json(ifs);",
            "    }",
            "#endif",
            "",
        ])

        lines.append(self.gen_json_helpers(indent_level=1))
        lines.append("")

        lines.append("};")
        lines.append("")

        lines.append(f"#endif // {guard_name}")
        out_dir = Path(include_dir) if include_dir is not None else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / file_name
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return str(out_path)

class DataField(DataSchema):
    """
    Represents a basic data field with a specific C++ type and bitwidth.
    These nodes are the leaves of the data schema tree and 
    correspond to actual data fields in the generated C++ struct.
    """
    def __init__(
            self, 
            name: str, 
            cpp_type: str,
            bitwidth: int, 
            description: Optional[str] = None):
        super().__init__(name, description)
        self.bitwidth = bitwidth
        self.cpp_type = cpp_type

    def get_bitwidth(self) -> int:
        return self.bitwidth
    
    def cpp_class_name(self) -> str:
        """For DataField, the C++ class name is just the cpp_type."""
        return self.cpp_type
    
    def gen_pack(self, indent_level: int = 0) -> str:
        # DataFields don't generate their own pack function; 
        # they are packed by their parent StructNode.
        return ""

    def gen_unpack(self, indent_level: int = 0) -> str:
        # DataFields don't generate their own unpack function;
        # they are unpacked by their parent StructNode.
        return ""
    
    def _gen_write_recursive(
        self, 
        word_bw: int, 
        dst_type: Literal["array", "stream", "axi4_stream"] = "array",
        target: Literal["x", "s"] = "x", 
        ipos0: int = 0, 
        iword0: int = 0, 
        prefix: str = "",
        params : Any = None
    ) -> Tuple[List[str], int, int]:
        """
        Implement the logic to write this field into the target interface.
        If word_bw < self.bitwidth, raise an error since we don't split individual 
        fields across words.  
        """
        if self.bitwidth > word_bw:
            raise ValueError(
                f"Field '{self.name}' with bitwidth {self.bitwidth} cannot fit into word_bw={word_bw}."
            )

        lines: List[str] = []
        curr_ipos = ipos0
        curr_iword = iword0

        if curr_ipos + self.bitwidth > word_bw:
            if dst_type == "stream":
                lines.append(f"    {target}.write(w);")
                lines.append("    w = 0;")
            elif dst_type == "axi4_stream":
                lines.append(f"     streamutils::write_axi4_word<{word_bw}>({target}, w, false);")
                lines.append("    w = 0;")

            curr_iword += 1
            curr_ipos = 0

        lhs = "w" if dst_type != "array" else f"{target}[{curr_iword}]"
        val_expr = self.to_uint_expr(prefix)

        if curr_ipos == 0 and self.bitwidth == word_bw:
            lines.append(f"    {lhs} = {val_expr};")
        else:
            if dst_type == "array" and curr_ipos == 0:
                lines.append(f"    {target}[{curr_iword}] = 0;")

            high = curr_ipos + self.bitwidth - 1
            low = curr_ipos
            lines.append(f"    {lhs}.range({high}, {low}) = {val_expr};")

        curr_ipos += self.bitwidth

        if curr_ipos == word_bw:
            if dst_type == "stream":
                lines.append(f"    {target}.write(w);")
                lines.append("    w = 0;")
            elif dst_type == "axi4_stream":
                lines.append(f"    streamutils::write_axi4_word<{word_bw}>({target}, w, false);")
                lines.append("    w = 0;")

            curr_iword += 1
            curr_ipos = 0

        return lines, curr_ipos, curr_iword

    def _gen_read_recursive(
        self,
        word_bw: int,
        src_type: Literal["array", "stream", "axi4_stream"] = "array",
        source: Literal["x", "s"] = "x",
        ipos0: int = 0,
        iword0: int = 0,
        prefix: str = "",
        params: Any = None,
    ) -> Tuple[List[str], int, int]:
        """Inverse of _gen_write_recursive: extract field bits from source words."""
        if self.bitwidth > word_bw:
            raise ValueError(
                f"Field '{self.name}' with bitwidth {self.bitwidth} cannot fit into word_bw={word_bw}."
            )

        lines: List[str] = []
        curr_ipos = ipos0
        curr_iword = iword0

        if curr_ipos + self.bitwidth > word_bw:
            curr_iword += 1
            curr_ipos = 0

        if src_type == "stream" and curr_ipos == 0:
            lines.append(f"    w = {source}.read();")
        elif src_type == "axi4_stream" and curr_ipos == 0:
            lines.append(f"    w = {source}.read().data;")

        if src_type == "array":
            word_expr = f"{source}[{curr_iword}]"
        else:
            word_expr = "w"

        if curr_ipos == 0 and self.bitwidth == word_bw:
            rhs_expr = word_expr
        else:
            high = curr_ipos + self.bitwidth - 1
            low = curr_ipos
            rhs_expr = f"{word_expr}.range({high}, {low})"

        assign_expr = self.from_uint_expr(rhs_expr)
        lines.append(f"    {prefix}{self.name} = {assign_expr};")

        curr_ipos += self.bitwidth
        if curr_ipos == word_bw:
            curr_iword += 1
            curr_ipos = 0

        return lines, curr_ipos, curr_iword

    def _nwords_per_inst_recursive(
        self,
        word_bw: int,
        ipos0: int = 0,
        iword0: int = 0,
    ) -> Tuple[int, int]:
        """
        Computes final (bit_pos, word_index) state after packing this field,
        matching the same boundary behavior used by _gen_write_recursive.
        """
        if self.bitwidth > word_bw:
            raise ValueError(
                f"Field '{self.name}' with bitwidth {self.bitwidth} cannot fit into word_bw={word_bw}."
            )

        curr_ipos = ipos0
        curr_iword = iword0

        if curr_ipos + self.bitwidth > word_bw:
            curr_iword += 1
            curr_ipos = 0

        curr_ipos += self.bitwidth

        if curr_ipos == word_bw:
            curr_iword += 1
            curr_ipos = 0

        return curr_ipos, curr_iword

    def _gen_dump_json_recursive(
        self,
        prefix: str,
        os_name: str,
        depth_expr: str,
        indent_var: str,
    ) -> List[str]:
        val_expr = f"{prefix}{self.name}"
        if isinstance(self, EnumField):
            val_expr = f"static_cast<int>({val_expr})"
        elif self.cpp_type.startswith("ap_uint"):
            val_expr = f"static_cast<unsigned long long>({val_expr})"
        elif self.cpp_type.startswith("ap_int"):
            val_expr = f"static_cast<long long>({val_expr})"

        return [f"{os_name} << {val_expr};"]

    def _gen_load_json_recursive(
        self,
        prefix: str,
        json_var: str,
        pos_var: str,
        ctx: str,
    ) -> List[str]:
        target = f"{prefix}{self.name}"

        if isinstance(self, FloatField):
            return [f"{target} = static_cast<{self.cpp_class_name()}>(_json_parse_number({json_var}, {pos_var}));"]

        if isinstance(self, EnumField):
            return [
                f"{target} = static_cast<{self.cpp_class_name()}>(static_cast<long long>(_json_parse_number({json_var}, {pos_var})));"
            ]

        if isinstance(self, IntField):
            if self.signed:
                return [
                    f"{target} = static_cast<{self.cpp_class_name()}>(static_cast<long long>(_json_parse_number({json_var}, {pos_var})));"
                ]
            return [
                f"{target} = static_cast<{self.cpp_class_name()}>(static_cast<unsigned long long>(_json_parse_number({json_var}, {pos_var})));"
            ]

        return [f"{target} = static_cast<{self.cpp_class_name()}>(_json_parse_number({json_var}, {pos_var}));"]


    def default_value(self) -> Any:
        return 0

    def gen_cpp_class(self) -> str:
        """
        Generates the C++ class definition via a typedef. For example, if cpp_type is "ap_int<16>", 
        this will generate:
            typedef ap_int<16> name;
        """
        return f"typedef    {self.cpp_type} {self.name};"

    def from_uint_expr(self, uint_expr: str) -> str:
        return f"({self.cpp_type})({uint_expr})"

    def is_close(
        self,
        other: DataSchema,
        rel_tol: float | None = None,
        abs_tol: float | None = 1e-8,
    ) -> bool:
        if not isinstance(other, self.__class__):
            return False

        v1 = self.val
        v2 = other.val

        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            return np.array_equal(np.asarray(v1), np.asarray(v2))

        return v1 == v2

    def _serialize_recursive(
        self,
        word_bw: int,
        words: List[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> Tuple[int, int]:
        if self.bitwidth > word_bw:
            raise ValueError(
                f"Field '{self.name}' with bitwidth {self.bitwidth} cannot fit into word_bw={word_bw}."
            )

        curr_ipos = ipos0
        curr_iword = iword0

        if curr_ipos + self.bitwidth > word_bw:
            curr_iword += 1
            curr_ipos = 0

        while len(words) <= curr_iword:
            words.append(0)

        current_val = self.val if self.val is not None else self.default_value()
        mask = (1 << self.bitwidth) - 1

        if isinstance(self, FloatField):
            if self.bitwidth == 32:
                field_bits = int(np.asarray(np.float32(current_val), dtype=np.float32).view(np.uint32))
            elif self.bitwidth == 64:
                field_bits = int(np.asarray(np.float64(current_val), dtype=np.float64).view(np.uint64))
            else:
                raise ValueError(f"Unsupported FloatField bitwidth={self.bitwidth} for serialization.")
        elif isinstance(self, EnumField):
            field_bits = int(current_val)
        elif isinstance(self, IntField) and self.bitwidth > 64 and isinstance(current_val, np.ndarray):
            field_bits = 0
            for idx, word in enumerate(current_val.astype(np.uint64)):
                field_bits |= int(word) << (64 * idx)
        else:
            field_bits = int(current_val)

        field_bits &= mask
        words[curr_iword] |= (field_bits << curr_ipos)

        curr_ipos += self.bitwidth
        if curr_ipos == word_bw:
            curr_iword += 1
            curr_ipos = 0

        return curr_ipos, curr_iword

    def _deserialize_recursive(
        self,
        word_bw: int,
        words: List[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> Tuple[int, int]:
        if self.bitwidth > word_bw:
            raise ValueError(
                f"Field '{self.name}' with bitwidth {self.bitwidth} cannot fit into word_bw={word_bw}."
            )

        curr_ipos = ipos0
        curr_iword = iword0

        if curr_ipos + self.bitwidth > word_bw:
            curr_iword += 1
            curr_ipos = 0

        if curr_iword >= len(words):
            word = 0
        else:
            word = words[curr_iword]

        mask = (1 << self.bitwidth) - 1
        field_bits = (word >> curr_ipos) & mask

        if isinstance(self, FloatField):
            if self.bitwidth == 32:
                value = np.asarray(np.uint32(field_bits), dtype=np.uint32).view(np.float32).item()
            elif self.bitwidth == 64:
                value = np.asarray(np.uint64(field_bits), dtype=np.uint64).view(np.float64).item()
            else:
                raise ValueError(f"Unsupported FloatField bitwidth={self.bitwidth} for deserialization.")
            self.val = value
        elif isinstance(self, EnumField):
            self.val = int(field_bits)
        else:
            self.val = int(field_bits)

        curr_ipos += self.bitwidth
        if curr_ipos == word_bw:
            curr_iword += 1
            curr_ipos = 0

        return curr_ipos, curr_iword

class IntField(DataField):
    """
    Represents an integer field with a specified bitwidth and signedness.

    Parameters
    ----------
    name : str
    bitwidth : int, optional
        The bitwidth of the integer field (default is 32).
    signed : bool, optional
        Whether the integer is signed (default is True).
    """
    def __init__(
            self, 
            name: str, 
            bitwidth: int = 32, 
            signed: bool = True,
            **kwargs):
        
        if signed:
            cpp_type = f"ap_int<{bitwidth}>"
        else:
            cpp_type = f"ap_uint<{bitwidth}>"
        super().__init__(name, cpp_type, bitwidth,**kwargs)
        self.signed = signed

    def to_uint_expr(self, prefix: Optional[str] = None) -> str:
        # HLS handles ap_int/int to ap_uint conversion natively via assignment
        if prefix is None:
            prefix = ""
        return f"{prefix}{self.name}"

    def _convert(self, new_val):
        if isinstance(new_val, (float, np.floating)) and not float(new_val).is_integer():
            raise ValueError(
                f"Cannot assign non-integer value {new_val} to IntField '{self.name}'."
            )

        if self.bitwidth <= 64:
            mask = (1 << self.bitwidth) - 1
            wrapped = int(new_val) & mask

            if self.signed:
                sign_bit = 1 << (self.bitwidth - 1)
                if wrapped & sign_bit:
                    wrapped -= (1 << self.bitwidth)
                return np.int32(wrapped) if self.bitwidth <= 32 else np.int64(wrapped)

            return np.uint32(wrapped) if self.bitwidth <= 32 else np.uint64(wrapped)

        num_words = math.ceil(self.bitwidth / 64)

        if isinstance(new_val, np.ndarray):
            arr = np.array(new_val, dtype=np.uint64)
            if arr.size != num_words:
                raise ValueError(
                    f"Array size mismatch for {self.name}: expected {num_words} words, got {arr.size}."
                )
            return arr

        if isinstance(new_val, (list, tuple)):
            arr = np.array(new_val, dtype=np.uint64)
            if arr.size != num_words:
                raise ValueError(
                    f"Array size mismatch for {self.name}: expected {num_words} words, got {arr.size}."
                )
            return arr

        mask = (1 << self.bitwidth) - 1
        wrapped = int(new_val) & mask
        arr = np.zeros(num_words, dtype=np.uint64)
        for i in range(num_words):
            arr[i] = np.uint64((wrapped >> (64 * i)) & 0xFFFFFFFFFFFFFFFF)
        return arr

    def default_value(self) -> Any:
        if self.bitwidth > 64:
            num_words = math.ceil(self.bitwidth / 64)
            return np.zeros(num_words, dtype=np.uint64)

        if self.signed:
            return np.int32(0) if self.bitwidth <= 32 else np.int64(0)

        return np.uint32(0) if self.bitwidth <= 32 else np.uint64(0)

   

class FloatField(DataField):
    def __init__(
            self, 
            name: str,
            bitwidth : int = 32,
            **kwargs):
        
        if bitwidth not in (32, 64):
            raise ValueError("FloatField only supports 32 or 64 bit widths.")
        cpp_type = "double" if bitwidth == 64 else "float"
        super().__init__(name=name, bitwidth=bitwidth, cpp_type=cpp_type,
                          **kwargs)

    def to_uint_expr(self, prefix: Optional[str] = None) -> str:
        """Floats require bit_cast to avoid value conversion."""
        if prefix is None:
            prefix = ""
        return f"streamutils::float_to_uint({prefix}{self.name})"

    def _convert(self, new_val):
        return np.float64(new_val) if self.bitwidth == 64 else np.float32(new_val)

    def default_value(self) -> Any:
        return np.float64(0.0) if self.bitwidth == 64 else np.float32(0.0)

    def from_uint_expr(self, uint_expr: str) -> str:
        if self.bitwidth != 32:
            raise ValueError("FloatField unpack currently supports only bitwidth=32.")
        return f"streamutils::uint_to_float((uint32_t)({uint_expr}))"

    def is_close(
        self,
        other: DataSchema,
        rel_tol: float | None = None,
        abs_tol: float | None = 1e-8,
    ) -> bool:
        if not isinstance(other, FloatField):
            return False

        if self.bitwidth != other.bitwidth:
            return False

        v1 = float(self.val)
        v2 = float(other.val)

        if rel_tol is None and abs_tol is None:
            return v1 == v2

        kwargs = {}
        if rel_tol is not None:
            kwargs["rtol"] = rel_tol
        if abs_tol is not None:
            kwargs["atol"] = abs_tol
        return bool(np.isclose(v1, v2, **kwargs))


class EnumField(DataField):
    def __init__(
            self,
            name: str,
            enum_type: Type[IntEnum],
            bitwidth: Optional[int] = None,
            default: Optional[IntEnum] = None,
            **kwargs):

        if not issubclass(enum_type, IntEnum):
            raise TypeError("EnumField requires enum_type to derive from IntEnum.")

        enum_values = [int(member.value) for member in enum_type]
        if any(value < 0 for value in enum_values):
            raise ValueError("EnumField currently supports only non-negative IntEnum values.")

        min_width = (max(enum_values).bit_length() or 1) if enum_values else 1
        if bitwidth is None:
            bitwidth = min_width
        elif bitwidth < min_width:
            raise ValueError(
                f"bitwidth={bitwidth} is too small for enum {enum_type.__name__}; needs at least {min_width} bits."
            )

        super().__init__(name=name, cpp_type=enum_type.__name__, bitwidth=bitwidth, **kwargs)
        self.enum_type = enum_type

        if default is None:
            default = list(enum_type)[0]
        self._default_member = self._coerce(default)

    def _coerce(self, value: Any) -> IntEnum:
        try:
            return self.enum_type(value)
        except ValueError as exc:
            raise ValueError(f"Value '{value}' is not a valid member of {self.enum_type.__name__}") from exc

    def _convert(self, new_val):
        return self._coerce(new_val)

    def to_uint_expr(self, prefix: Optional[str] = None) -> str:
        if prefix is None:
            prefix = ""
        return f"(ap_uint<{self.bitwidth}>)({prefix}{self.name})"

    def default_value(self) -> Any:
        return self._default_member

    def from_uint_expr(self, uint_expr: str) -> str:
        return f"({self.enum_type.__name__})({uint_expr})"

    def gen_cpp_class(self) -> str:
        """
        Generates a C++ enum definition for this EnumField's enum_type, 
        along with a typedef for the field itself. For example:

            enum OpCode {
                READ = 0,
                WRITE = 1,
                SYNC = 2,
            };
        """
        enum_lines = [f"enum {self.enum_type.__name__} {{"]
        for member in self.enum_type:
            enum_lines.append(f"    {member.name} = {member.value},")
        enum_lines.append("};")
        return "\n".join(enum_lines)

    def is_close(
        self,
        other: DataSchema,
        rel_tol: float | None = None,
        abs_tol: float | None = 1e-8,
    ) -> bool:
        if not isinstance(other, EnumField):
            return False

        if self.enum_type is not other.enum_type:
            return False

        return self.val == other.val

    

class DataList(DataSchema):
    """
    Represents a collection of DataSchemas that correspond to fields in a C++ class.  This is
    the main way to define structured data types that will be mapped to C++ classes 
    with multiple fields.
    """
    def __init__(
        self, 
        name: str, 
        elements: List[DataSchema] = None, 
        description: Optional[str] = None
    ):
        """
        Constructor

        Parameters
        ---------
        name : str
            The name of this data list, used for C++ class naming and code generation.
        elements : List[DataSchema], optional
            A list of DataSchema instances that are the fields/elements of this data list.
            You can also add elements later using the add_elem() method.
        description : Optional[str]
            An optional human-readable description of this data list, 
            useful for documentation and code comments.
        """
        super().__init__(name, description)
        self.elements = []
        self._element_map = {}
        if elements is None:
            elements = []
        for elem in elements:
            self.add_elem(elem)

    def add_elem(self, element: DataSchema):
        """
        Adds a DataSchema element to this DataList.

        This will also set the element as an attribute of this DataList instance, allowing
        you to access it as self.element_name.  The element's name must be unique among the 
        existing attributes.
        """
        self.elements.append(element)

        if not element.name:
            raise ValueError("Elements must have names.")
        
        # Check against existing standard attributes (like 'name', 'elements')
        if element.name in self.__dict__:
            raise ValueError(f"Name '{element.name}' conflicts with internal attribute.")
        self._element_map[element.name] = element
        element.val = element.default_value()
       

    def __getattr__(self, name):
        """
        Redirect attribute access to internal elements if needed.

        Note that __getattr__ is only called if the normal attribute lookup fails, 
        so this won't interfere with standard attributes like 'name' or 'elements'.  
        It will only be triggered for names that aren't already defined as attributes, 
        which is where we want to look up in the _element_map.
        For example, if you have a DataList `ComplexSample` with an element nameed `r`,
        you can access it as complex_sample.r, which will return the current 
        value of the `r` element.
        """
        if name in self._element_map:
            element = self._element_map[name]
            if isinstance(element, DataList):
                return element
            return element.val
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Intercepts all assignments."""
        # Use __dict__ to check for initialization state safely
        if "_element_map" in self.__dict__ and name in self._element_map:
            # This triggers the child Field's _convert and hardware constraints
            self._element_map[name].val = value
        else:
            # Normal attribute setting (name, elements, etc.)
            super().__setattr__(name, value)

    @property
    def val(self) -> dict:
        """The 'Snapshot' of the whole struct."""
        return {el.name: el.val for el in self.elements}

    @val.setter
    def val(self, new_val):
        """Bulk update from a dictionary or from another instance of the same DataList type."""
        if isinstance(new_val, self.__class__):
            new_val = new_val.val

        if not isinstance(new_val, dict):
            raise TypeError("DataList.val must be a dictionary or matching DataList instance.")

        for key, value in new_val.items():
            if key in self._element_map:
                self._element_map[key].val = value

    def _convert(self, new_val):
        if isinstance(new_val, self.__class__):
            return new_val

        if isinstance(new_val, dict):
            inst = self.default_value()
            inst.val = new_val
            return inst

        raise TypeError(
            f"Cannot assign value of type {type(new_val).__name__} to DataList '{self.__class__.__name__}'."
        )

    def default_value(self):
        """Returns a typed default instance for this struct."""
        return self.__class__(name=self.name)
    

    def get_bitwidth(self) -> int:
        """Sum of bitwidths of all children."""
        return sum(node.get_bitwidth() for node in self.elements)

    def gen_pack(self, indent_level: int = 0) -> str:
        """
        Generates the static pack_to_uint method for this collection.
        It iterates through elements and slices them into the wide ap_uint.
        """
        cls_name = self.cpp_class_name()
        total_bits = self.get_bitwidth()
        indent = self._get_indent(indent_level)
        inner_indent = self._get_indent(indent_level + 1)

        lines = [
            f"{indent}static ap_uint<bitwidth> pack_to_uint(const {cls_name}& data) {{",
            f"{inner_indent}ap_uint<bitwidth> res = 0;"
        ]

        current_lsb = 0
        for node in self.elements:
            width = node.get_bitwidth()
            # We use the node's to_uint_expr, but we must prefix it with 'data.'
            # because the static method takes 'const C& data' as an argument.
            expr = node.to_uint_expr(prefix="data.") 
            high = current_lsb + width - 1
            low = current_lsb
            lines.append(f"{inner_indent}res.range({high}, {low}) = {expr};")
            current_lsb += width

        lines.append(f"{inner_indent}return res;")
        lines.append(f"{indent}}}")
        
        return "\n".join(lines)

    def gen_unpack(self, indent_level: int = 0) -> str:
        """
        Generates the static unpack_from_uint method for this collection.
        It slices the packed ap_uint and reconstructs each element.
        """
        cls_name = self.cpp_class_name()
        total_bits = self.get_bitwidth()
        indent = self._get_indent(indent_level)
        inner_indent = self._get_indent(indent_level + 1)

        lines = [
            f"{indent}static {cls_name} unpack_from_uint(const ap_uint<bitwidth>& packed) {{",
            f"{inner_indent}{cls_name} data;"
        ]

        current_lsb = 0
        for node in self.elements:
            width = node.get_bitwidth()
            high = current_lsb + width - 1
            low = current_lsb
            slice_expr = f"packed.range({high}, {low})"
            rhs_expr = node.from_uint_expr(slice_expr)
            lines.append(f"{inner_indent}data.{node.name} = {rhs_expr};")
            current_lsb += width

        lines.append(f"{inner_indent}return data;")
        lines.append(f"{indent}}}")

        return "\n".join(lines)

    def gen_cpp_class(self) -> str:
        """
        Generates the C++ class definition for this DataList and its children.

        class ClassName {
        public:
            // Child elements as fields
            ChildType1 child1;
            ChildType2 child2;
            ...    
        }

        """
        class_name = self.cpp_class_name()
        lines: List[str] = [
            f"class {class_name} {{",
            "public:",
        ]

        for element in self.elements:
            elem_class = element.cpp_class_name()
            lines.append(f"    {elem_class} {element.name};")

        lines.extend([
            "",
            f"    static constexpr int bitwidth = {self.get_bitwidth()};",
            "",
            "};",
        ])

        return "\n".join(lines)

    def _gen_write_recursive(
        self, 
        word_bw: int, 
        dst_type: Literal["array", "stream", "axi4_stream"] = "array",
        target: Literal["x", "s"] = "x", 
        ipos0: int = 0, 
        iword0: int = 0, 
        prefix: str = "",
        params : Any = None
    ) -> Tuple[List[str], int, int]:
        """
        TODO:  Implement the logic to write lines for all the child elements,
        recurisvely calling their _gen_write_recursive methods and correctly updating
        and passing the bit position and word index as needed.
        """
        lines: List[str] = []
        curr_ipos = ipos0
        curr_iword = iword0

        for element in self.elements:
            elem_prefix = prefix
            if isinstance(element, DataList) and element.name:
                elem_prefix = f"{prefix}{element.name}."

            elem_lines, curr_ipos, curr_iword = element._gen_write_recursive(
                word_bw=word_bw,
                dst_type=dst_type,
                target=target,
                ipos0=curr_ipos,
                iword0=curr_iword,
                prefix=elem_prefix,
                params=params
            )
            lines.extend(elem_lines)

        return lines, curr_ipos, curr_iword

    def _gen_read_recursive(
        self,
        word_bw: int,
        src_type: Literal["array", "stream", "axi4_stream"] = "array",
        source: Literal["x", "s"] = "x",
        ipos0: int = 0,
        iword0: int = 0,
        prefix: str = "",
        params: Any = None,
    ) -> Tuple[List[str], int, int]:
        """Recursive inverse read generation for all child elements."""
        lines: List[str] = []
        curr_ipos = ipos0
        curr_iword = iword0

        for element in self.elements:
            elem_prefix = prefix
            if isinstance(element, DataList) and element.name:
                elem_prefix = f"{prefix}{element.name}."

            elem_lines, curr_ipos, curr_iword = element._gen_read_recursive(
                word_bw=word_bw,
                src_type=src_type,
                source=source,
                ipos0=curr_ipos,
                iword0=curr_iword,
                prefix=elem_prefix,
                params=params,
            )
            lines.extend(elem_lines)

        return lines, curr_ipos, curr_iword

    def _nwords_per_inst_recursive(
        self,
        word_bw: int,
        ipos0: int = 0,
        iword0: int = 0,
    ) -> Tuple[int, int]:
        curr_ipos = ipos0
        curr_iword = iword0

        for element in self.elements:
            curr_ipos, curr_iword = element._nwords_per_inst_recursive(
                word_bw=word_bw,
                ipos0=curr_ipos,
                iword0=curr_iword,
            )

        return curr_ipos, curr_iword

    def _gen_dump_json_recursive(
        self,
        prefix: str,
        os_name: str,
        depth_expr: str,
        indent_var: str,
    ) -> List[str]:
        lines: List[str] = [f"{os_name} << \"{{\";"]

        if self.elements:
            lines.append(f"{os_name} << \"\\n\";")

        for idx, element in enumerate(self.elements):
            lines.append(f"for (int i = 0; i < ({depth_expr} + 1) * {indent_var}; ++i) {{ {os_name} << ' '; }}")
            lines.append(f"{os_name} << \"\\\"{element.name}\\\": \";")

            elem_prefix = prefix
            if isinstance(element, DataList) and element.name:
                elem_prefix = f"{prefix}{element.name}."

            lines.extend(
                element._gen_dump_json_recursive(
                    prefix=elem_prefix,
                    os_name=os_name,
                    depth_expr=f"{depth_expr} + 1",
                    indent_var=indent_var,
                )
            )

            if idx < len(self.elements) - 1:
                lines.append(f"{os_name} << \",\";")
            lines.append(f"{os_name} << \"\\n\";")

        if self.elements:
            lines.append(f"for (int i = 0; i < ({depth_expr}) * {indent_var}; ++i) {{ {os_name} << ' '; }}")

        lines.append(f"{os_name} << \"}}\";")
        return lines

    def _gen_load_json_recursive(
        self,
        prefix: str,
        json_var: str,
        pos_var: str,
        ctx: str,
    ) -> List[str]:
        lines: List[str] = [
            f"_json_expect_char({json_var}, {pos_var}, '{{');",
        ]

        seen_flags = [f"seen_{ctx}_{element.name}" for element in self.elements]
        for flag in seen_flags:
            lines.append(f"bool {flag} = false;")

        lines.extend([
            "bool first = true;",
            "while (true) {",
            f"    _json_skip_ws({json_var}, {pos_var});",
            f"    if ({pos_var} < {json_var}.size() && {json_var}[{pos_var}] == '}}') {{",
            f"        ++{pos_var};",
            "        break;",
            "    }",
            "    if (!first) {",
            f"        _json_expect_char({json_var}, {pos_var}, ',');",
            "    }",
            "    first = false;",
            f"    std::string key = _json_parse_string({json_var}, {pos_var});",
            f"    _json_expect_char({json_var}, {pos_var}, ':');",
        ])

        for idx, element in enumerate(self.elements):
            cond = "if" if idx == 0 else "else if"
            elem_prefix = prefix
            if isinstance(element, DataList) and element.name:
                elem_prefix = f"{prefix}{element.name}."

            lines.append(f"    {cond} (key == \"{element.name}\") {{")
            lines.append(f"        seen_{ctx}_{element.name} = true;")

            child_ctx = f"{ctx}_{element.name}"
            child_lines = element._gen_load_json_recursive(
                prefix=elem_prefix,
                json_var=json_var,
                pos_var=pos_var,
                ctx=child_ctx,
            )
            for child_line in child_lines:
                lines.append(f"        {child_line}")

            lines.append("    }")

        lines.extend([
            "    else {",
            "        throw std::runtime_error(\"Malformed JSON: unexpected key for schema.\");",
            "    }",
            "}",
        ])

        for element in self.elements:
            lines.extend([
                f"if (!seen_{ctx}_{element.name}) {{",
                f"    throw std::runtime_error(\"Malformed JSON: missing required key '{element.name}'.\");",
                "}",
            ])

        return lines

    def _serialize_recursive(
        self,
        word_bw: int,
        words: List[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> Tuple[int, int]:
        curr_ipos = ipos0
        curr_iword = iword0

        for element in self.elements:
            curr_ipos, curr_iword = element._serialize_recursive(
                word_bw=word_bw,
                words=words,
                ipos0=curr_ipos,
                iword0=curr_iword,
            )

        return curr_ipos, curr_iword

    def _deserialize_recursive(
        self,
        word_bw: int,
        words: List[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> Tuple[int, int]:
        curr_ipos = ipos0
        curr_iword = iword0

        for element in self.elements:
            curr_ipos, curr_iword = element._deserialize_recursive(
                word_bw=word_bw,
                words=words,
                ipos0=curr_ipos,
                iword0=curr_iword,
            )

        return curr_ipos, curr_iword

    def is_close(
        self,
        other: DataSchema,
        rel_tol: float | None = None,
        abs_tol: float | None = 1e-8,
    ) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if len(self.elements) != len(other.elements):
            return False

        for element in self.elements:
            if element.name not in other._element_map:
                return False

            if not element.is_close(other._element_map[element.name], rel_tol=rel_tol, abs_tol=abs_tol):
                return False

        return True
    
class DataArray(DataSchema):
    """
    Represents an array of DataSchemas. This can be used for both fixed and dynamic
    sized arrays. 
    """
    def __init__(
        self, 
        name: str, 
        element_type: DataSchema, 
        max_shape: List[int] = None, 
        static: bool = True,
        description: Optional[str] = None
    ):
        """
        Parameters
        ----------
        name : str
            The name of this data array, used for C++ class naming and code generation.
        element_type : DataSchema
            The DataSchema type of the elements in this array. This defines the type of each element and how they are packed.
        max_shape : List[int], optional
            The maximum shape of the array. For a 1D array, this would be [max_length]. For a 2D array, this would be [max_rows, max_cols], etc. This is used for code generation and validation.
        static : bool, optional
            Whether this array has a static shape (True) or dynamic shape (False). If static
            is True, the generated C++ code will assume a fixed size defined by max_shape. If False, the generated code will need to take additional parameters for the actual shape and may include bounds checks.
        description : str, optional
            An optional human-readable description of this data array, useful for documentation and code comments.
        
        """
        super().__init__(name, description)
        self.element_type = element_type
        self.max_shape = max_shape
        self.static = static