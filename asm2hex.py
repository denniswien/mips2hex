#!/usr/bin/env python3
"""
MIPS Assembly to Intel HEX Converter
Converts MIPS assembly instructions to Intel HEX format
Supports .data and .text sections like MARS
"""

import re
import sys
from typing import Dict, List, Tuple, Optional


class MipsAssembler:
    def __init__(self):
        # MIPS register mappings
        self.registers = {
            '$zero': 0, '$0': 0,
            '$at': 1, '$1': 1,
            '$v0': 2, '$2': 2, '$v1': 3, '$3': 3,
            '$a0': 4, '$4': 4, '$a1': 5, '$5': 5, '$a2': 6, '$6': 6, '$a3': 7, '$7': 7,
            '$t0': 8, '$8': 8, '$t1': 9, '$9': 9, '$t2': 10, '$10': 10, '$t3': 11, '$11': 11,
            '$t4': 12, '$12': 12, '$t5': 13, '$13': 13, '$t6': 14, '$14': 14, '$t7': 15, '$15': 15,
            '$s0': 16, '$16': 16, '$s1': 17, '$17': 17, '$s2': 18, '$18': 18, '$s3': 19, '$19': 19,
            '$s4': 20, '$20': 20, '$s5': 21, '$21': 21, '$s6': 22, '$22': 22, '$s7': 23, '$23': 23,
            '$t8': 24, '$24': 24, '$t9': 25, '$25': 25,
            '$k0': 26, '$26': 26, '$k1': 27, '$27': 27,
            '$gp': 28, '$28': 28, '$sp': 29, '$29': 29, '$fp': 30, '$30': 30, '$ra': 31, '$31': 31
        }

        # MIPS instruction opcodes and function codes
        self.r_type_functs = {
            'add': 0x20, 'addu': 0x21, 'and': 0x24, 'div': 0x1A, 'divu': 0x1B,
            'mult': 0x18, 'multu': 0x19, 'nor': 0x27, 'or': 0x25, 'sll': 0x00,
            'sllv': 0x04, 'slt': 0x2A, 'sltu': 0x2B, 'sra': 0x03, 'srav': 0x07,
            'srl': 0x02, 'srlv': 0x06, 'sub': 0x22, 'subu': 0x23, 'xor': 0x26,
            'jr': 0x08, 'jalr': 0x09, 'mfhi': 0x10, 'mflo': 0x12, 'mthi': 0x11, 'mtlo': 0x13
        }

        self.i_type_opcodes = {
            'addi': 0x08, 'addiu': 0x09, 'andi': 0x0C, 'beq': 0x04, 'bne': 0x05,
            'lbu': 0x24, 'lhu': 0x25, 'll': 0x30, 'lui': 0x0F, 'lw': 0x23, 'lb': 0x20,
            'ori': 0x0D, 'sb': 0x28, 'sc': 0x38, 'sh': 0x29, 'slti': 0x0A,
            'sltiu': 0x0B, 'sw': 0x2B, 'xori': 0x0E, 'lh': 0x21
        }

        self.j_type_opcodes = {
            'j': 0x02, 'jal': 0x03
        }

        self.special2_functs = {
            'mul': 0x02
        }

        self.labels = {}
        self.data_labels = {}
        self.data_segment = []  # List of (address, data_bytes)
        self.text_segment = []  # List of (address, instruction)

        # Memory layout
        self.data_start = 0x00000000
        self.text_start = 0x00000000
        self.current_data_address = self.data_start
        self.current_text_address = self.text_start

    def encode_special2(self, rs: int, rt: int, rd: int, funct: int) -> int:
        """Encode SPECIAL2-format R-type instruction (opcode 0x1C)"""
        return (0x1C << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (funct & 0x3F)
    def fits_signed16(self, value: int) -> bool:
        """Check if a value fits in 16-bit signed integer"""
        return -32768 <= value <= 32767

    def parse_register(self, reg_str: str) -> int:
        """Parse register string and return register number"""
        reg_str = reg_str.strip().replace(',', '')
        if reg_str in self.registers:
            return self.registers[reg_str]
        raise ValueError(f"Invalid register: {reg_str}")

    def parse_immediate(self, imm_str: str) -> int:
        imm_str = imm_str.split("#")[0].strip()  # Strip inline comments

        """Parse immediate value (decimal or hex)"""
        imm_str = imm_str.strip().replace(',', '')
        if imm_str.startswith('0x'):
            return int(imm_str, 16)
        return int(imm_str)

    def encode_r_type(self, opcode: int, rs: int, rt: int, rd: int, shamt: int, funct: int) -> int:
        """Encode R-type instruction"""
        return (opcode << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (shamt << 6) | funct

    def encode_i_type(self, opcode: int, rs: int, rt: int, immediate: int) -> int:
        """Encode I-type instruction"""
        immediate = immediate & 0xFFFF  # 16-bit immediate
        return (opcode << 26) | (rs << 21) | (rt << 16) | immediate

    def encode_j_type(self, opcode: int, address: int) -> int:
        """Encode J-type instruction"""
        address = (address >> 2) & 0x3FFFFFF  # 26-bit word address
        return (opcode << 26) | address

    def parse_data_directive(self, line: str) -> Optional[List[int]]:
        """Parse data directives like .word, .byte, .ascii, etc."""
        line = line.strip()
        if not line.startswith('.'):
            return None

        parts = line.split(None, 1)
        directive = parts[0].lower()

        if len(parts) < 2:
            return []

        data_str = parts[1].strip()
        data_bytes = []

        try:
            if directive == '.word':
                # Parse comma-separated 32-bit words
                values = [x.strip() for x in data_str.split(',')]
                for val in values:
                    if val.startswith('0x'):
                        word = int(val, 16)
                    else:
                        word = int(val)
                    # Store as little-endian bytes
                    data_bytes.extend([
                        word & 0xFF,
                        (word >> 8) & 0xFF,
                        (word >> 16) & 0xFF,
                        (word >> 24) & 0xFF
                    ])

            elif directive == '.half' or directive == '.halfword':
                # Parse comma-separated 16-bit halfwords
                values = [x.strip() for x in data_str.split(',')]
                for val in values:
                    if val.startswith('0x'):
                        halfword = int(val, 16)
                    else:
                        halfword = int(val)
                    # Store as little-endian bytes
                    data_bytes.extend([
                        halfword & 0xFF,
                        (halfword >> 8) & 0xFF
                    ])

            elif directive == '.byte':
                # Parse comma-separated bytes
                values = [x.strip() for x in data_str.split(',')]
                for val in values:
                    if val.startswith('0x'):
                        byte = int(val, 16)
                    else:
                        byte = int(val)
                    data_bytes.append(byte & 0xFF)

            elif directive == '.ascii':
                # Parse ASCII string (without null terminator)
                if data_str.startswith('"') and data_str.endswith('"'):
                    string = data_str[1:-1]
                    # Handle escape sequences
                    string = string.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                    string = string.replace('\\"', '"').replace('\\\\', '\\')
                    data_bytes.extend([ord(c) for c in string])

            elif directive == '.asciiz':
                # Parse ASCII string with null terminator
                if data_str.startswith('"') and data_str.endswith('"'):
                    string = data_str[1:-1]
                    # Handle escape sequences
                    string = string.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                    string = string.replace('\\"', '"').replace('\\\\', '\\')
                    data_bytes.extend([ord(c) for c in string])
                    data_bytes.append(0)  # Null terminator

            elif directive == '.space':
                # Allocate space (fill with zeros)
                size = int(data_str)
                data_bytes.extend([0] * size)

            elif directive == '.align':
                # Alignment directive - pad to alignment boundary
                alignment = int(data_str)
                current_pos = self.current_data_address
                aligned_pos = ((current_pos + alignment - 1) // alignment) * alignment
                padding = aligned_pos - current_pos
                data_bytes.extend([0] * padding)

            return data_bytes

        except Exception as e:
            print(f"Error parsing data directive '{line}': {e}")
            return None

    def process_data_section(self, lines: List[str]) -> None:
        """Process the .data section"""
        in_data_section = False

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.lower() == '.data':
                in_data_section = True
                continue
            elif line.lower() == '.text':
                in_data_section = False
                continue

            if not in_data_section:
                continue

            # Handle labels in data section
            if ':' in line:
                label, rest = line.split(':', 1)
                label = label.strip()
                self.data_labels[label] = self.current_data_address
                line = rest.strip()
                if not line:
                    continue

            # Parse data directive
            data_bytes = self.parse_data_directive(line)
            if data_bytes is not None:
                if data_bytes:  # Only add if there are actual bytes
                    self.data_segment.append((self.current_data_address, data_bytes))
                    self.current_data_address += len(data_bytes)

    def process_text_section(self, lines: List[str]) -> None:
        """Process the .text section"""
        in_text_section = False

        # First pass: collect labels
        address = self.current_text_address
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.lower() == '.text':
                in_text_section = True
                continue
            elif line.lower() == '.data':
                in_text_section = False
                continue

            if not in_text_section:
                continue

            if ':' in line:
                label = line.split(':')[0].strip()
                self.labels[label] = address

            # Count instructions (not labels or directives)
            if not line.startswith('.') and not ((':' in line) and (line.split(':', 1)[1].strip() == '')):
                address += 4

        # Second pass: assemble instructions
        address = self.current_text_address
        in_text_section = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.lower() == '.text':
                in_text_section = True
                continue
            elif line.lower() == '.data':
                in_text_section = False
                continue

            if not in_text_section:
                continue

            instruction_word = self.assemble_instruction(line, address)
            if instruction_word is not None:
                self.text_segment.append((address, instruction_word))
                address += 4

    def assemble_instruction(self, instruction: str, address: int) -> Optional[int]:
        # Handle pseudo-instructions
        if instruction.startswith("li "):
            reg, imm = instruction[3:].split(",", 1)
            reg = reg.strip()
            imm_val = self.parse_immediate(imm.strip())
            if self.fits_signed16(imm_val):
                return self.encode_i_type(self.i_type_opcodes["addi"], 0, self.parse_register(reg), imm_val)
            else:
                upper = (imm_val >> 16) & 0xFFFF
                lower = imm_val & 0xFFFF
                rt = self.parse_register(reg)
                self.text_segment.append((address, self.encode_i_type(self.i_type_opcodes["lui"], 0, rt, upper)))
                self.text_segment.append((address + 4, self.encode_i_type(self.i_type_opcodes["ori"], rt, rt, lower)))
                return None

        elif instruction.startswith("la "):
            reg, label = instruction[3:].split(",", 1)
            reg = reg.strip()
            label = label.strip()
            addr = self.data_labels.get(label, 0)
            upper = (addr >> 16) & 0xFFFF
            lower = addr & 0xFFFF
            rt = self.parse_register(reg)
            self.text_segment.append((address, self.encode_i_type(self.i_type_opcodes["lui"], 0, rt, upper)))
            self.text_segment.append((address + 4, self.encode_i_type(self.i_type_opcodes["ori"], rt, rt, lower)))
            return None

        elif instruction.startswith("move "):
            rd, rs = instruction[5:].split(",", 1)
            rd = self.parse_register(rd.strip())
            rs = self.parse_register(rs.strip())
            return self.encode_r_type(0, rs, 0, rd, 0, self.r_type_functs["or"])


        """Assemble a single MIPS instruction"""
        instruction = instruction.strip()
        if not instruction or instruction.startswith('#') or instruction.startswith('.'):
            return None

        # Handle labels
        if ':' in instruction:
            label, rest = instruction.split(':', 1)
            instruction = rest.strip()
            if not instruction:
                return None

        parts = re.split(r'[,\s]+', instruction)
        op = parts[0].lower()

        try:
            if op in self.special2_functs:  # e.g.  mul rd, rs, rt
                rd = self.parse_register(parts[1])
                rs = self.parse_register(parts[2])
                rt = self.parse_register(parts[3])
                return self.encode_special2(rs, rt, rd,
                                            self.special2_functs[op])
            # R-type instructions
            elif op in self.r_type_functs:
                if op in ['sll', 'srl', 'sra']:  # Shift instructions
                    rd = self.parse_register(parts[1])
                    rt = self.parse_register(parts[2])
                    shamt = self.parse_immediate(parts[3])
                    return self.encode_r_type(0, 0, rt, rd, shamt, self.r_type_functs[op])
                elif op in ['jr']:
                    rs = self.parse_register(parts[1])
                    return self.encode_r_type(0, rs, 0, 0, 0, self.r_type_functs[op])
                elif op in ['mfhi', 'mflo']:
                    rd = self.parse_register(parts[1])
                    return self.encode_r_type(0, 0, 0, rd, 0, self.r_type_functs[op])
                else:  # Standard R-type
                    rd = self.parse_register(parts[1])
                    rs = self.parse_register(parts[2])
                    rt = self.parse_register(parts[3])
                    return self.encode_r_type(0, rs, rt, rd, 0, self.r_type_functs[op])

            # I-type instructions
            elif op in self.i_type_opcodes:
                if op in ['lw', 'sw', 'lb', 'sb', 'lh', 'sh', 'lbu', 'lhu']:  # Memory instructions
                    rt = self.parse_register(parts[1])
                    # Parse offset(base) format or label
                    mem_part = parts[2]
                    if '(' in mem_part:
                        offset_str, base_str = mem_part.split('(')
                        if offset_str in self.data_labels:
                            offset = self.data_labels[offset_str]
                        else:
                            offset = self.parse_immediate(offset_str) if offset_str else 0
                        base = self.parse_register(base_str.replace(')', ''))
                    else:
                        # Could be a label or direct address
                        if mem_part in self.data_labels:
                            offset = self.data_labels[mem_part]
                            base = 0  # $zero
                        else:
                            offset = 0
                            base = self.parse_register(mem_part)
                    return self.encode_i_type(self.i_type_opcodes[op], base, rt, offset)
                elif op in ['beq', 'bne']:  # Branch instructions
                    rs = self.parse_register(parts[1])
                    rt = self.parse_register(parts[2])
                    # Handle label or immediate
                    target = parts[3]
                    if target in self.labels:
                        offset = (self.labels[target] - address - 4) // 4
                    else:
                        offset = self.parse_immediate(target)
                    return self.encode_i_type(self.i_type_opcodes[op], rs, rt, offset)
                elif op == 'lui':
                    rt = self.parse_register(parts[1])
                    immediate = self.parse_immediate(parts[2])
                    return self.encode_i_type(self.i_type_opcodes[op], 0, rt, immediate)
                else:  # Standard I-type
                    rt = self.parse_register(parts[1])
                    rs = self.parse_register(parts[2])
                    immediate = self.parse_immediate(parts[3])
                    return self.encode_i_type(self.i_type_opcodes[op], rs, rt, immediate)

            # J-type instructions
            elif op in self.j_type_opcodes:
                target = parts[1]
                if target in self.labels:
                    addr = self.labels[target]
                elif target in self.data_labels:
                    addr = self.data_labels[target]
                else:
                    addr = self.parse_immediate(target)
                return self.encode_j_type(self.j_type_opcodes[op], addr)

            else:
                print(f"Warning: Unknown instruction '{op}'")
                return None

        except Exception as e:
            print(f"Error assembling instruction '{instruction}': {e}")
            return None


def write_data_record(hex_lines: List[str], address: int, data_bytes: List[int]) -> None:
    """Write a data record to the hex lines"""
    byte_count = len(data_bytes)
    record_type = 0x00  # Data record
    record = f":{byte_count:02X}{address:04X}{record_type:02X}"
    for byte in data_bytes:
        record += f"{byte:02X}"

    checksum = calculate_checksum(record[1:])
    hex_lines.append(f"{record}{checksum:02X}")


def generate_intel_hex(text_segments: List[Tuple[int, int]]) -> List[str]:
    """Generate Intel HEX with instructions at 0x000, 0x100, 0x200... and 1 per line"""
    hex_lines = []

    for i, (_, instruction) in enumerate(text_segments):
        address = i * 4

        # Convert to big-endian bytes
        bytes_ = [
            (instruction >> 24) & 0xFF,
            (instruction >> 16) & 0xFF,
            (instruction >> 8) & 0xFF,
            instruction & 0xFF
        ]

        # Format Intel HEX line
        record = f":04{address:04X}00" + ''.join(f"{b:02X}" for b in bytes_)
        checksum = calculate_checksum(record[1:])
        hex_lines.append(f"{record}{checksum:02X}")

    # End of file
    hex_lines.append(":00000001FF")
    return hex_lines



def calculate_checksum(hex_string: str) -> int:
    """Calculate Intel HEX checksum"""
    total = 0
    for i in range(0, len(hex_string), 2):
        total += int(hex_string[i:i + 2], 16)
    return (-total) & 0xFF


def main():
    if len(sys.argv) != 3:
        print("Usage: python mips_to_hex.py <input.asm> <output.hex>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    assembler = MipsAssembler()

    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()

        assembler.process_data_section(lines)  # Needed for label resolution
        assembler.process_text_section(lines)

        # Only emit instruction segment (no data)
        hex_lines = generate_intel_hex(assembler.text_segment)

        with open(output_file, 'w') as f:
            for line in hex_lines:
                f.write(line + '\n')

        print("Output written to:", output_file)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)



if __name__ == "__main__":
    main()
