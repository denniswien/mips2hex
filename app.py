import streamlit as st
from asm2hex import MipsAssembler, generate_intel_hex

def convert_asm_to_hex(asm_code: str) -> str:
    assembler = MipsAssembler()
    lines = asm_code.splitlines()
    assembler.process_data_section(lines)
    assembler.process_text_section(lines)
    hex_lines = generate_intel_hex(assembler.text_segment)
    return '\n'.join(hex_lines)

st.title("MIPS → Intel-HEX converter")

asm_source = st.text_area("Paste MIPS assembly:", height=300)

if st.button("Convert"):
    try:
        hex_out = convert_asm_to_hex(asm_source)
        st.success("✓ Conversion successful")
        st.text_area("Intel HEX:", hex_out, height=300)
        st.download_button("Download .hex", hex_out, file_name="output.hex")
    except ValueError as err:
        st.error(str(err))
