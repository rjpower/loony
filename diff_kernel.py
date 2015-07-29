# knl.temporary_variables: a dict name->"array-like"
# knl.args: a list (.name)

import loopy as lp
import islpy as isl
from pymbolic.mapper.differentiator import DifferentiationMapper
from loopy.symbolic import IdentityMapperMixin
import pymbolic.primitives as p
var = p.Variable
from loopy.isl_helpers import make_slab

# There's another thing in loopy called "substitution rules" which
# make this a little more complicated. I've kept that out of this for
# now to avoid confusing you. But we should talk about it sometime--
# it's mostly orthogonal to this effort anyway.


def func_map(i, func, args):
    if func.name == "tanh":
        return (1 - var("tanh")(*args))**2
    else:
        raise NotImplementedError("derivative of '%s'" % func.name)


class LoopyDiffMapper(DifferentiationMapper, IdentityMapperMixin):
    def __init__(self, by_name, tgt_index):
        self.by_name = by_name
        self.tgt_index = tgt_index
        self.function_map = func_map

    def map_variable(self, expr):
        if expr.name == self.by_name:
            assert len(self.tgt_index) == 0
            return 1

        else:
            # Assume everything else is data that needs to be kept
            return expr

    map_tagged_variable = map_variable

    def map_subscript(self, expr):
        if expr.aggregate.name == self.by_name:
            index = expr.index
            if not isinstance(expr.index, tuple):
                index = (expr.index,)

            assert len(self.tgt_index) == len(index)

            conds = [
                p.Comparison(var(ti), "==", ei)
                for ti, ei in zip(self.tgt_index, index)
                ]

            if len(conds) == 1:
                and_conds, = conds
            elif len(conds) > 1:
                and_conds = p.LogicalAnd(tuple(conds))
            else:
                assert False

            return p.If(and_conds, 1, 0)

        else:
            return type(expr)(expr.aggregate, self.rec(expr.index))


def diff_kernel(knl, outputs, by, diff_iname_prefix="diff_i"):
    by_arg = knl.arg_dict[by]
    additional_shape = by_arg.shape

    diff_inames = [
        knl.get_var_name_generator()(diff_iname_prefix+str(i))
        for i in range(len(additional_shape))]

    # modify shape of output variables
    new_args = []
    for arg in knl.args:
        if arg.name not in outputs:
            new_args.append(arg)
            continue

        new_args.append(
            lp.GlobalArg(
                arg.name,
                arg.dtype,
                shape=arg.shape + additional_shape,
                dim_tags=arg.dim_tags + ("c",) * len(additional_shape),
            ))

    # FIXME: be more selective what domains to bolt this onto
    new_domains = []
    for dom in knl.domains:
        base_idx = dom.dim(isl.dim_type.set)
        dom = dom.insert_dims(isl.dim_type.set, base_idx, len(diff_inames))
        for i, diff_iname in enumerate(diff_inames):
            dom = dom.set_dim_name(isl.dim_type.set, base_idx+i, diff_iname)
            dom = dom & make_slab(
                dom.space, diff_iname, 0, additional_shape[i])

        new_domains.append(dom)

    diff_mapper = LoopyDiffMapper(by, tuple(diff_inames))

    new_insns = []
    for insn in knl.instructions:
        (lhs_name, lhs_ind), = insn.assignees_and_indices()

        new_insn = insn.copy(
            assignee=var(lhs_name)[lhs_ind + (var(diff_iname),)],
            expression=diff_mapper(insn.expression)
            )
        new_insns.append(new_insn)

    return knl.copy(
        args=new_args,
        domains=new_domains,
        instructions=new_insns
        )

# FIXME: batching
# FIXME: multiple levels of dependencies
