from nolane_ai.reasoning.cps import CanonicalProblemState, TableConstraint, Variable
from nolane_ai.reasoning.propagation import propagate
from nolane_ai.reasoning.search import EpisodeNogoodStore, solve_branch, solve_hybrid
from nolane_ai.reasoning.worlds import make_structure_dense_world, make_conflict_world


def test_gac_propagates_unary_anchor_through_equality_chain():
    problem = CanonicalProblemState(
        variables=(Variable("a", (0, 1)), Variable("b", (0, 1)), Variable("c", (0, 1))),
        constraints=(
            TableConstraint("ab", ("a", "b"), ((0, 0), (1, 1))),
            TableConstraint("bc", ("b", "c"), ((0, 0), (1, 1))),
            TableConstraint("anchor", ("c",), ((1,),)),
        ),
    )
    result = propagate(problem)
    assert result.consistent
    assert result.domains == {"a": (1,), "b": (1,), "c": (1,)}
    assert result.stats.values_pruned == 3


def test_hybrid_uses_fewer_accounted_operations_on_structure_dense_world():
    problem = make_structure_dense_world(seed=17, variables=7, domain_size=3)
    branch = solve_branch(problem)
    hybrid = solve_hybrid(problem)
    assert branch.verified and hybrid.verified
    assert hybrid.accounted_operations < branch.accounted_operations
    assert problem.is_solution(hybrid.solution)


def test_oracle_core_variable_priority_reduces_unsat_search_cost():
    problem = make_conflict_world(seed=9, decoys=5)
    chronological = solve_branch(problem)
    oracle = solve_branch(problem, priority_variables=problem.oracle_conflict_variables)
    assert chronological.verified and oracle.verified
    assert chronological.solution is None and oracle.solution is None
    assert oracle.accounted_operations < chronological.accounted_operations


def test_episode_local_nogood_prunes_repeated_dead_end_across_restarts():
    problem = make_conflict_world(seed=11, decoys=3)
    store = EpisodeNogoodStore()
    first = solve_branch(problem, nogood_store=store)
    second = solve_branch(problem, nogood_store=store)
    assert first.solution is None and second.solution is None
    assert second.stats.nogood_hits > 0
    assert second.accounted_operations < first.accounted_operations


def test_satisfiable_backtracking_world_has_dead_end_and_nogoods_do_not_cover_valid_solution():
    from nolane_ai.reasoning.worlds import make_satisfiable_backtracking_world
    problem = make_satisfiable_backtracking_world(seed=13)
    solutions = list(problem.enumerate_assignments(limit=1024))
    valid = [assignment for assignment in solutions if problem.is_solution(assignment)]
    assert valid
    store = EpisodeNogoodStore()
    first = solve_branch(problem, nogood_store=store)
    assert first.solution is not None
    assert first.stats.dead_end_signatures
    assert len(store) > 0
    assert all(not store.matches(solution) for solution in valid)
