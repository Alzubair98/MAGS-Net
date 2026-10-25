"""Import only the audited model definitions, without running notebook experiments.

The publication module has executable dataset/training/evaluation cells at module
scope. A normal import would execute those cells. This adapter compiles only its
seven named definitions, checks their AST fingerprint, and supplies torch/nn/F.
It does not copy, replace, or alter the architecture.
"""
import ast
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from magsnet_demo.config import MANIFEST, MODEL_SOURCE
from magsnet_demo.utils import DemoError

MODEL_DEFINITIONS = (
    "conv_bn_relu", "Stem", "MAGBlock", "MAGSStage", "SegmentationDecoder",
    "ClassificationHead", "MAGSNet",
)


def model_nodes(source):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in MODEL_DEFINITIONS]
    if [n.name for n in nodes] != list(MODEL_DEFINITIONS):
        raise DemoError("The publication model definitions are missing or have changed.")
    return nodes


def definition_fingerprint(nodes):
    # Python 3.12 adds an empty type_params field; retain the audited 3.10 encoding.
    content = {n.name: ast.dump(n, include_attributes=False).replace(", type_params=[]", "") for n in nodes}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def load_model_class():
    if not MODEL_SOURCE.is_file():
        raise DemoError("Model definition missing. Keep this demo inside the MAGS-Net project "
                        "or set MAGSNET_PROJECT_ROOT as explained in README.md.")
    nodes = model_nodes(MODEL_SOURCE.read_text(encoding="utf-8"))
    if definition_fingerprint(nodes) != MANIFEST["model_definition_sha256"]:
        raise DemoError("The architecture differs from the audited publication model. "
                        "Restore the matching source before running inference.")
    namespace = {"torch": torch, "nn": nn, "F": F}
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(MODEL_SOURCE), "exec")
    exec(code, namespace)
    return namespace["MAGSNet"]
