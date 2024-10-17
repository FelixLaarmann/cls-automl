from typing import Any

from clsp import (
    DSL,
    Constructor,
    LVar,
)
from clsp.types import Literal
class CFG_Repository:

    def __init__(self):
        #nonterminals
        self.d2 = Constructor("D2")
        self.d1 = Constructor("D1")
        self.d0 = Constructor("D0")
        self.d = Constructor("D")
        self.c = Constructor("C")
        self.cl = Constructor("CL")
        self.op = Constructor("OP")
        self.convblock = Constructor("CONVBLOCK")
        self.act = Constructor("ACT")
        self.conv = Constructor("CONV")
        self.norm = Constructor("NORM")
        #terminals
        #self.down = Constructor("DOWN")

    def delta(self) -> dict[str, list[Any]]:
        return {}

    def gamma(self):
        return {
            "Sequential3_D2_1": DSL().
            Use("x", self.d1).
            Use("y", self.d1).
            Use("z", self.d0).
            In(self.d2),
            "Sequential3_D2_2": DSL().
            Use("x", self.d0).
            Use("y", self.d1).
            Use("z", self.d1).
            In(self.d2),
            "Sequential4_D2":  DSL().
            Use("w", self.d1).
            Use("x", self.d1).
            Use("y", self.d0).
            Use("z", self.d0).
            In(self.d2),
            "Sequential3_D1": DSL().
            Use("x", self.c).
            Use("y", self.c).
            Use("z", self.d).
            In(self.d1),
            "Sequential4_D1": DSL().
            Use("w", self.c).
            Use("x", self.c).
            Use("y", self.c).
            Use("z", self.d).
            In(self.d1),
            "Residual3_D1": DSL().
            Use("w", self.c).
            Use("x", self.c).
            Use("y", self.d).
            Use("z", self.d).
            In(self.d1),
            "Sequential3_D0": DSL().
            Use("x", self.c).
            Use("y", self.c).
            Use("z", self.cl).
            In(self.d0),
            "Sequential4_D0": DSL().
            Use("w", self.c).
            Use("x", self.c).
            Use("y", self.c).
            Use("z", self.cl).
            In(self.d0),
            "Residual3_D0": DSL().
            Use("w", self.c).
            Use("x", self.c).
            Use("y", self.cl).
            Use("z", self.cl).
            In(self.d0),
            #"down": self.down,
            "Sequential2_D": DSL().
            Use("x", self.cl).
            #Use("y", self.down).
            In(self.d),
            "Sequential3_D": DSL().
            Use("x", self.cl).
            Use("y", self.cl).
            #Use("z", self.down).
            In(self.d),
            "Residual2_D": DSL().
            Use("x", self.cl).
            #Use("y", self.down).
            #Use("z", self.down).
            In(self.d),
            "Sequential2_C": DSL().
            Use("x", self.cl).
            Use("y", self.cl).
            In(self.c),
            "Sequential3_C": DSL().
            Use("x", self.cl).
            Use("y", self.cl).
            Use("z", self.cl).
            In(self.c),
            "Residual2_C": DSL().
            Use("x", self.cl).
            Use("y", self.cl).
            Use("z", self.cl).
            In(self.c),
            "Cell": DSL().
            Use("u", self.op).
            Use("v", self.op).
            Use("w", self.op).
            Use("x", self.op).
            Use("y", self.op).
            Use("z", self.op).
            In(self.cl),
            "zero": self.op,
            "id": self.op,
            "avg_pool": self.op,
            "CONVBLOCK_OP": DSL().
            Use("x", self.convblock).
            In(self.op),
            "Sequential3_CONVBLOCK": DSL().
            Use("x", self.act).
            Use("y", self.conv).
            Use("z", self.norm).
            In(self.convblock),
            "relu": self.act,
            "hardswish": self.act,
            "mish": self.act,
            "conv1x1": self.conv,
            "conv3x3": self.conv,
            "dconv3x3": self.conv,
            "batch": self.norm,
            "instance": self.norm,
            "layer": self.norm,
        }

