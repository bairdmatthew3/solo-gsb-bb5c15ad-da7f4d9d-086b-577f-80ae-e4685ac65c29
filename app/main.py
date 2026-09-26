"""FastAPI 应用入口。"""

from __future__ import annotations

from fastapi import FastAPI

from .models import (
    AssignmentIn,
    AssignmentOut,
    AssignRequest,
    AssignResponse,
    AssignRolesRequest,
    AssignRolesResponse,
    BottleneckCutEdgeOut,
    BottleneckCutOut,
    LoadRatioOut,
    LockedConflictOut,
    RoleAssignmentOut,
    RoleGapOut,
    RoleShortfallOut,
    SharedPoolOut,
    UnsatisfiedOut,
    WorkloadBalanceOut,
)
from .solver import (
    BottleneckCut,
    ManuscriptSpec,
    ReviewerSpec,
    solve_assignment,
    solve_role_assignment,
)

app = FastAPI(
    title="Anonymous Review Assignment API",
    version="1.5.0",
    description=(
        "匿名评审分配服务：基于全局最大流（非逐稿贪心）求满足主题能力、"
        "利益冲突与容量约束的完整分配；不可行时返回缺口诊断与全局最小割证明"
        "（bottleneck_cut）且不输出部分结果。"
        "支持分赛道容量：稿件可声明 track，评审人可声明 track_capacities，"
        "赛道限额与总容量在同一流网络中联立求解。"
        "POST /assign-roles 支持多评审角色：稿件声明各角色所需人数、评审人声明"
        "可担任角色，全部稿件与角色在同一流网络中一次全局求解，保证同一评审人对"
        "同一稿件至多承担一个角色，避免按角色分别分配时争用共享容量。"
        "支持增量重排：携带 previous_assignments / locked_assignments 时，"
        "锁定关系经稿件需求、评审资格与容量校验后强制保留，其余名额在全局可行"
        "前提下最少改动地复用上一轮分配。"
        "支持负载均衡：balance_workload=true 时完整可行解按字典序优化——"
        "先最小化最大负载率（分配数/capacity，约分分数比较，不依赖浮点），"
        "再最小化相对上一轮的改动数，最后按 ID 破同。"
        "服务无状态，不持久化任何稿件/评审信息。"
    ),
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "anonymous-review-assign",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


def _to_pair_set(
    items: list[AssignmentIn] | None,
) -> frozenset[tuple[str, str]] | None:
    """把 AssignmentIn 列表展开为 (manuscript_id, reviewer_id) 对集合（自动去重）。"""
    if items is None:
        return None
    return frozenset(
        (a.manuscript_id, rid) for a in items for rid in set(a.reviewer_ids)
    )


def _to_bottleneck_cut_out(cut: BottleneckCut) -> BottleneckCutOut:
    def edges(group) -> list[BottleneckCutEdgeOut]:
        return [BottleneckCutEdgeOut(**vars(e)) for e in group]

    return BottleneckCutOut(
        source_manuscripts=list(cut.source_manuscripts),
        source_reviewer_tracks=[list(pair) for pair in cut.source_reviewer_tracks],
        source_reviewers=list(cut.source_reviewers),
        demand=edges(cut.demand),
        eligibility=edges(cut.eligibility),
        track_capacity=edges(cut.track_capacity),
        reviewer_capacity=edges(cut.reviewer_capacity),
        max_assignable=cut.max_assignable,
        shortfall=cut.shortfall,
    )


@app.post("/assign", response_model=AssignResponse, tags=["assign"])
def assign(req: AssignRequest) -> AssignResponse:
    default_required = req.required_reviewers
    manuscripts = [
        ManuscriptSpec(
            id=m.id,
            authors=tuple(m.authors),
            topics=tuple(m.topics),
            required_reviewers=(
                m.required_reviewers
                if m.required_reviewers is not None
                else default_required
            ),
            track=m.track,
        )
        for m in req.manuscripts
    ]
    reviewers = [
        ReviewerSpec(
            id=r.id,
            topics=tuple(r.topics),
            capacity=r.capacity,
            conflicted_authors=tuple(r.conflicted_authors),
            conflicted_manuscripts=tuple(r.conflicted_manuscripts),
            track_capacities=(
                tuple(sorted(r.track_capacities.items()))
                if r.track_capacities
                else ()
            ),
        )
        for r in req.reviewers
    ]

    result = solve_assignment(
        manuscripts,
        reviewers,
        previous=_to_pair_set(req.previous_assignments),
        locked=_to_pair_set(req.locked_assignments),
        balance_workload=req.balance_workload,
    )

    # 锁定关系冲突：结构化原因，不输出任何部分分配
    if result.locked_conflicts:
        return AssignResponse(
            status="infeasible",
            total_required=result.total_required,
            total_assigned=result.total_assigned,
            assignments=None,
            unsatisfied=None,
            shared_pool=None,
            locked_conflicts=[
                LockedConflictOut(**vars(c)) for c in result.locked_conflicts
            ],
        )

    if result.feasible:
        return AssignResponse(
            status="feasible",
            total_required=result.total_required,
            total_assigned=result.total_assigned,
            assignments=[
                AssignmentOut(manuscript_id=mid, reviewer_ids=list(rids))
                for mid, rids in sorted(result.assignments.items())
            ],
            unsatisfied=None,
            shared_pool=None,
            locked_conflicts=None,
            balance=(
                WorkloadBalanceOut(
                    max_load_ratio=LoadRatioOut(
                        numerator=result.balance.max_load_ratio_num,
                        denominator=result.balance.max_load_ratio_den,
                    ),
                    changes=result.balance.changes,
                )
                if result.balance
                else None
            ),
        )

    return AssignResponse(
        status="infeasible",
        total_required=result.total_required,
        total_assigned=result.total_assigned,
        assignments=None,
        unsatisfied=[UnsatisfiedOut(**vars(u)) for u in result.unsatisfied],
        shared_pool=SharedPoolOut(**result.shared_pool) if result.shared_pool else None,
        locked_conflicts=None,
        bottleneck_cut=(
            _to_bottleneck_cut_out(result.bottleneck_cut)
            if result.bottleneck_cut
            else None
        ),
    )


@app.post("/assign-roles", response_model=AssignRolesResponse, tags=["assign"])
def assign_roles(req: AssignRolesRequest) -> AssignRolesResponse:
    manuscripts = [
        ManuscriptSpec(
            id=m.id,
            authors=tuple(m.authors),
            topics=tuple(m.topics),
            required_reviewers=sum(m.role_requirements.values()),
            track=m.track,
            role_requirements=tuple(sorted(m.role_requirements.items())),
        )
        for m in req.manuscripts
    ]
    reviewers = [
        ReviewerSpec(
            id=r.id,
            topics=tuple(r.topics),
            capacity=r.capacity,
            conflicted_authors=tuple(r.conflicted_authors),
            conflicted_manuscripts=tuple(r.conflicted_manuscripts),
            track_capacities=(
                tuple(sorted(r.track_capacities.items()))
                if r.track_capacities
                else ()
            ),
            roles=tuple(sorted(set(r.roles))),
        )
        for r in req.reviewers
    ]

    result = solve_role_assignment(manuscripts, reviewers)

    if result.feasible:
        assignments = [
            RoleAssignmentOut(manuscript_id=mid, role=role, reviewer_ids=list(rids))
            for (mid, role), rids in sorted(result.assignments.items())
        ]
        return AssignRolesResponse(
            status="feasible",
            total_required=result.total_required,
            total_assigned=result.total_assigned,
            assignments=assignments,
            role_shortfalls=None,
            role_gaps=None,
        )

    # 不可行：按角色汇总缺口（missing>0 的角色才出现），不输出任何部分分配
    gap_acc: dict[str, dict[str, int]] = {}
    for s in result.shortfalls:
        acc = gap_acc.setdefault(s.role, {"required": 0, "assigned": 0, "missing": 0})
        acc["required"] += s.required
        acc["assigned"] += s.assigned
        acc["missing"] += s.missing
    role_gaps = [
        RoleGapOut(
            role=role,
            required=acc["required"],
            assigned=acc["assigned"],
            missing=acc["missing"],
        )
        for role, acc in sorted(gap_acc.items())
    ]

    return AssignRolesResponse(
        status="infeasible",
        total_required=result.total_required,
        total_assigned=result.total_assigned,
        assignments=None,
        role_shortfalls=[
            RoleShortfallOut(**vars(s)) for s in sorted(
                result.shortfalls, key=lambda s: (s.manuscript_id, s.role)
            )
        ],
        role_gaps=role_gaps,
    )
