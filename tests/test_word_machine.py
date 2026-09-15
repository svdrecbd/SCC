"""Generic VM edge semantics for the bounded construction screen."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.screen_word_machine import (
    ADD, CALL, DEC, HALT, LOAD, RET, Machine, edit_command, ins,
)


def machine_with(code):
    return Machine(code + [0] * (64 - len(code)))


def test_unsigned_wrap_and_instruction_count():
    m = machine_with([ins(DEC, 0), ins(ADD, 1, 0, 0), ins(HALT)])
    assert m.execute(3)[4] == 3
    assert m.ram[48:50] == [65535, 65534]


def test_call_return_charges_every_instruction():
    m = machine_with([ins(CALL) | 2, ins(HALT), ins(DEC, 0), ins(RET)])
    assert m.execute(4)[4] == 4
    assert m.ram[59] == 1 and m.ram[48] == 65535


def test_step_limit_and_invalid_address():
    with pytest.raises(TimeoutError):
        machine_with([ins(CALL)]).execute(5)
    with pytest.raises(ValueError, match="LOAD outside"):
        machine_with([ins(LOAD, 0, 0, 15)]).execute(1)


def test_edit_interface_exposes_data_registers_and_pc():
    m = machine_with([ins(HALT)])
    for address in range(65):
        m.edit(edit_command(address, 65535))
        assert (m.ram[address] if address < 64 else m.pc) == 65535
    with pytest.raises(ValueError):
        m.edit(edit_command(65, 0))
