import streamlit as st
from asm2hex import MipsAssembler, generate_intel_hex  # your existing classes/functions

def convert_asm_to_hex(asm_code: str) -> str:
    assembler = MipsAssembler()
    lines = asm_code.splitlines()
    assembler.process_data_section(lines)
    assembler.process_text_section(lines)
    hex_lines = generate_intel_hex(assembler.text_segment)
    return '\n'.join(hex_lines)

# Streamlit UI
st.title("MIPS to Intel HEX Converter")

asm_input = st.text_area("Paste your MIPS Assembly Code here:", height=300)

if st.button("Convert"):
    try:
        hex_output = convert_asm_to_hex(asm_input)
        st.success("Conversion successful!")
        st.text_area("Intel HEX Output:", hex_output, height=300)
    except Exception as e:
        st.error(f"Error during conversion: {e}")
