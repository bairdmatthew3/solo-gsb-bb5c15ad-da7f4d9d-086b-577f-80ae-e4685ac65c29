"""POST /assign-roles 的端到端测试。"""

from __future__ import annotations

import random
from itertools import combinations, product

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _role_map(response: dict) -> dict[str, dict[str, list[str]]]:
    """响应 -> {manuscript_id: {role: [reviewer_id]}}。"""
    return {
        a["manuscript_id"]: {r["role"]: r["reviewer_ids"] for r in a["roles"]}
        for a in response["assignments"]
    }


def _verify_role_constraints(body: dict, response: dict) -> None:
    """校验分角色分配满足全部约束：各角色人数、一人一稿一角、容量、赛道、主题、冲突。"""
    assert response["status"] == "feasible"
    # 输出顺序确定：稿件按 ID 排序，角色按名称排序，名单按 ID 排序
    assert [a["manuscript_id"] for a in response["assignments"]] == sorted(
        a["manuscript_id"] for a in response["assignments"]
    )
    assignments = _role_map(response)
    reviewers = {r["id"]: r for r in body["reviewers"]}

    used = {rid: 0 for rid in reviewers}
    track_used: dict[str, dict[str, int]] = {rid: {} for rid in reviewers}
    for a in response["assignments"]:
        role_names = [r["role"] for r in a["roles"]]
        assert role_names == sorted(role_names), "角色未按名称排序"
    for m in body["manuscripts"]:
        per_role = assignments[m["id"]]
        # 每个声明的角色都有名单（需求为 0 的角色给空名单）
        assert set(per_role) == set(m["role_requirements"])
        flat: list[str] = []
        track = m.get("track")
        for role, n in m["role_requirements"].items():
            rids = per_role[role]
            assert len(rids) == n, f"{m['id']} 角色 {role} 人数不足"
            assert rids == sorted(rids), "名单未按 ID 排序"
            assert len(set(rids)) == len(rids), "同一角色重复分配评审人"
            flat.extend(rids)
            for rid in rids:
                r = reviewers[rid]
                assert role in r.get("roles", []), "评审人不能担任该角色"
                assert rid not in m["authors"], "自稿冲突"
                assert not set(m["authors"]) & set(r.get("conflicted_authors", [])), \
                    "利益冲突"
                assert m["id"] not in r.get("conflicted_manuscripts", [])
                if m["topics"]:
                    assert set(m["topics"]) & set(r["topics"]), "主题不匹配"
                used[rid] += 1
                if track is not None:
                    per = track_used[rid]
                    per[track] = per.get(track, 0) + 1
        # 同一评审人对同一稿件最多承担一个角色
        assert len(set(flat)) == len(flat), "同一评审人在同一稿件身兼多角"
    for rid, n in used.items():
        assert n <= reviewers[rid]["capacity"], f"{rid} 超出总容量"
    for rid, per_track in track_used.items():
        caps = reviewers[rid].get("track_capacities") or {}
        for track, n in per_track.items():
            if track in caps:
                assert n <= caps[track], f"{rid} 赛道 {track} 超出限额"


# ---------- 可行场景 ----------

def test_roles_feasible_grouped_and_sorted():
    """可行：按稿件、角色分组的完整名单，顺序确定。"""
    body = {
        "manuscripts": [
            {"id": "m2", "authors": ["bob"], "topics": ["db"],
             "role_requirements": {"reviewer": 2}},
            {"id": "m1", "authors": ["alice"], "topics": ["ml"],
             "role_requirements": {"reviewer": 1, "chair": 1}},
        ],
        "reviewers": [
            {"id": "r3", "topics": ["db"], "capacity": 2, "roles": ["reviewer"]},
            {"id": "r1", "topics": ["ml"], "capacity": 2,
             "roles": ["chair", "reviewer"]},
            {"id": "r2", "topics": ["ml", "db"], "capacity": 2,
             "roles": ["reviewer"]},
            {"id": "r4", "topics": ["ml", "db"], "capacity": 2,
             "roles": ["reviewer"]},
        ],
    }
    resp = client.post("/assign-roles", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "feasible"
    assert data["total_required"] == 4
    assert data["total_assigned"] == 4
    assert data["unsatisfied"] is None
    _verify_role_constraints(body, data)
    # 确定性结果：chair 只能由 r1 担任；m1.reviewer 不能再用 r1（一人一稿一角）
    assert _role_map(data) == {
        "m1": {"chair": ["r1"], "reviewer": ["r2"]},
        "m2": {"reviewer": ["r2", "r3"]},
    }


def test_roles_global_solve_not_per_role_greedy():
    """按角色分别分配会争用共享容量而失败，全局一次求解成功。

    r1 可担任 chair/reviewer 但容量 1，r2 只能担任 chair。
    逐角色贪心（chair 优先、按 ID）会把 m1 的 chair 分给 r1，m2 的 reviewer
    便无人可担任；全局解必须让 r1 去当 m2 的 reviewer、m1 的 chair 改由 r2 担任。
    """
    body = {
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"],
             "role_requirements": {"chair": 1}},
            {"id": "m2", "authors": [], "topics": ["t"],
             "role_requirements": {"reviewer": 1}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 1,
             "roles": ["chair", "reviewer"]},
            {"id": "r2", "topics": ["t"], "capacity": 1, "roles": ["chair"]},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "feasible"
    _verify_role_constraints(body, data)
    assert _role_map(data) == {
        "m1": {"chair": ["r2"]},
        "m2": {"reviewer": ["r1"]},
    }


def test_roles_capacity_shared_across_roles_and_manuscripts():
    """评审人总容量不分角色：所有角色、所有稿件共享同一个容量。"""
    body = {
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"],
             "role_requirements": {"chair": 1}},
            {"id": "m2", "authors": [], "topics": ["t"],
             "role_requirements": {"reviewer": 1}},
            {"id": "m3", "authors": [], "topics": ["t"],
             "role_requirements": {"reviewer": 1}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 2,
             "roles": ["chair", "reviewer"]},
            {"id": "r2", "topics": ["t"], "capacity": 2, "roles": ["reviewer"]},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "feasible"
    assert data["total_assigned"] == 3
    _verify_role_constraints(body, data)


def test_roles_track_capacity_applies():
    """分赛道限额对分角色分配同样生效（与总容量在同一网络联立求解）。"""
    body = {
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"], "track": "A",
             "role_requirements": {"reviewer": 1}},
            {"id": "m2", "authors": [], "topics": ["t"], "track": "A",
             "role_requirements": {"reviewer": 1}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 5, "roles": ["reviewer"],
             "track_capacities": {"A": 1}},
            {"id": "r2", "topics": ["t"], "capacity": 1, "roles": ["reviewer"]},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "feasible"
    _verify_role_constraints(body, data)
    # r1 的 A 赛道限额 1：两篇 A 稿至多一篇分给 r1
    assert _role_map(data) == {
        "m1": {"reviewer": ["r2"]},
        "m2": {"reviewer": ["r1"]},
    }


def test_roles_conflicts_and_topics_apply():
    """主题匹配与利益冲突规则与 /assign 一致。"""
    body = {
        "manuscripts": [
            {"id": "m1", "authors": ["alice"], "topics": ["ml"],
             "role_requirements": {"reviewer": 1}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["ml"], "capacity": 1, "roles": ["reviewer"],
             "conflicted_authors": ["alice"]},
            {"id": "r2", "topics": ["db"], "capacity": 1, "roles": ["reviewer"]},
            {"id": "r3", "topics": ["ml"], "capacity": 1, "roles": ["reviewer"]},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "feasible"
    _verify_role_constraints(body, data)
    assert data["assignments"] == [
        {"manuscript_id": "m1",
         "roles": [{"role": "reviewer", "reviewer_ids": ["r3"]}]}
    ]


def test_roles_empty_requirements_feasible():
    """稿件不声明任何角色需求：可行，roles 为空列表。"""
    body = {
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"], "role_requirements": {}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 1, "roles": ["chair"]},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "feasible"
    assert data["total_required"] == 0
    assert data["assignments"] == [{"manuscript_id": "m1", "roles": []}]


def test_roles_zero_requirement_role_listed_empty():
    """需求为 0 的角色出现在名单中（空名单），保证输出结构完整确定。"""
    body = {
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"],
             "role_requirements": {"chair": 0, "reviewer": 1}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 1, "roles": ["reviewer"]},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "feasible"
    assert data["assignments"] == [
        {"manuscript_id": "m1", "roles": [
            {"role": "chair", "reviewer_ids": []},
            {"role": "reviewer", "reviewer_ids": ["r1"]},
        ]}
    ]


# ---------- 一人一稿一角 ----------

def test_roles_same_reviewer_at_most_one_role_per_manuscript():
    """同一评审人对同一稿件最多承担一个角色（容量再大也不能身兼两角）。"""
    body = {
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"],
             "role_requirements": {"chair": 1, "reviewer": 1}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 5,
             "roles": ["chair", "reviewer"]},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "infeasible"
    assert data["assignments"] is None
    assert data["total_required"] == 2
    assert data["total_assigned"] == 1
    unsat = data["unsatisfied"]
    assert len(unsat) == 1
    # 角色按名称排序，chair 先满足，reviewer 缺 1
    assert unsat[0]["manuscript_id"] == "m1"
    assert unsat[0]["role"] == "reviewer"
    assert unsat[0]["required"] == 1
    assert unsat[0]["assigned"] == 0
    assert unsat[0]["missing"] == 1
    assert unsat[0]["eligible_reviewers"] == 1
    assert unsat[0]["eligible_remaining_capacity"] == 4


def test_roles_multi_role_reviewer_takes_one_role_per_manuscript():
    """可担任多角的评审人在同一稿件上只占一角，但可在不同稿件担任不同角色。"""
    body = {
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"],
             "role_requirements": {"chair": 1, "reviewer": 1}},
            {"id": "m2", "authors": [], "topics": ["t"],
             "role_requirements": {"chair": 1, "reviewer": 1}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 2,
             "roles": ["chair", "reviewer"]},
            {"id": "r2", "topics": ["t"], "capacity": 2,
             "roles": ["chair", "reviewer"]},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "feasible"
    _verify_role_constraints(body, data)


# ---------- 不可行场景 ----------

def test_roles_infeasible_reports_per_role_gap():
    """不可行：按 (稿件, 角色) 给出缺口，不输出任何部分分配。"""
    body = {
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"],
             "role_requirements": {"chair": 1, "reviewer": 2}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 2, "roles": ["reviewer"]},
            {"id": "r2", "topics": ["t"], "capacity": 2, "roles": ["reviewer"]},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "infeasible"
    assert data["assignments"] is None
    assert data["total_required"] == 3
    assert data["total_assigned"] == 2
    unsat = data["unsatisfied"]
    assert len(unsat) == 1
    u = unsat[0]
    assert u["manuscript_id"] == "m1"
    assert u["role"] == "chair"
    assert u["required"] == 1
    assert u["assigned"] == 0
    assert u["missing"] == 1
    assert u["eligible_reviewers"] == 0
    assert u["eligible_remaining_capacity"] == 0


def test_roles_infeasible_capacity_shared_with_other_role():
    """共享容量被另一角色占满导致本角色缺口：缺口只报给真正缺人的角色。"""
    body = {
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"],
             "role_requirements": {"chair": 1}},
            {"id": "m2", "authors": [], "topics": ["t"],
             "role_requirements": {"reviewer": 2}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 1,
             "roles": ["chair", "reviewer"]},
            {"id": "r2", "topics": ["t"], "capacity": 1, "roles": ["reviewer"]},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "infeasible"
    assert data["assignments"] is None
    unsat = {(u["manuscript_id"], u["role"]): u for u in data["unsatisfied"]}
    assert set(unsat) == {("m2", "reviewer")}
    assert unsat[("m2", "reviewer")]["missing"] == 1


def test_roles_reviewer_without_roles_never_assigned():
    """未声明可担任角色的评审人（roles 缺省）不参与任何分配。"""
    body = {
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"],
             "role_requirements": {"reviewer": 1}},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 5},
        ],
    }
    data = client.post("/assign-roles", json=body).json()
    assert data["status"] == "infeasible"
    assert data["assignments"] is None
    assert data["unsatisfied"][0]["eligible_reviewers"] == 0


# ---------- 确定性 / 稳定性 ----------

def _random_role_instance(seed: int) -> dict:
    rng = random.Random(seed)
    topics = ["ml", "nlp", "cv"]
    roles_pool = ["chair", "reviewer", "meta"]
    authors = [f"author{i}" for i in range(10)]
    manuscripts = [
        {
            "id": f"m{i}",
            "authors": rng.sample(authors, rng.randint(0, 2)),
            "topics": rng.sample(topics, rng.randint(1, 2)),
            "role_requirements": {
                role: rng.randint(1, 2)
                for role in rng.sample(roles_pool, rng.randint(1, 2))
            },
            **({"track": rng.choice(["A", "B"])} if rng.random() < 0.4 else {}),
        }
        for i in range(6)
    ]
    reviewers = [
        {
            "id": f"r{j}",
            "topics": rng.sample(topics, rng.randint(1, 3)),
            "capacity": rng.randint(1, 4),
            "roles": rng.sample(roles_pool, rng.randint(1, 2)),
            "conflicted_authors": rng.sample(authors, rng.randint(0, 2)),
            **(
                {"track_capacities": {
                    t: rng.randint(0, 2) for t in rng.sample(["A", "B"], 1)
                }}
                if rng.random() < 0.4 else {}
            ),
        }
        for j in range(8)
    ]
    return {"manuscripts": manuscripts, "reviewers": reviewers}


def test_roles_same_input_same_output():
    body = _random_role_instance(seed=42)
    r1 = client.post("/assign-roles", json=body).json()
    r2 = client.post("/assign-roles", json=body).json()
    assert r1 == r2


def test_roles_input_order_does_not_change_result():
    body = _random_role_instance(seed=7)
    shuffled = {
        "manuscripts": list(reversed(body["manuscripts"])),
        "reviewers": list(reversed(body["reviewers"])),
    }
    r1 = client.post("/assign-roles", json=body).json()
    r2 = client.post("/assign-roles", json=shuffled).json()
    assert r1 == r2


# ---------- 请求校验 ----------

def test_roles_duplicate_manuscript_id_rejected():
    body = {
        "manuscripts": [
            {"id": "m1", "role_requirements": {}},
            {"id": "m1", "role_requirements": {}},
        ],
        "reviewers": [],
    }
    resp = client.post("/assign-roles", json=body)
    assert resp.status_code == 422


def test_roles_duplicate_reviewer_id_rejected():
    body = {
        "manuscripts": [],
        "reviewers": [
            {"id": "r1", "capacity": 1},
            {"id": "r1", "capacity": 2},
        ],
    }
    resp = client.post("/assign-roles", json=body)
    assert resp.status_code == 422


def test_roles_negative_requirement_rejected():
    body = {
        "manuscripts": [
            {"id": "m1", "role_requirements": {"reviewer": -1}},
        ],
        "reviewers": [],
    }
    resp = client.post("/assign-roles", json=body)
    assert resp.status_code == 422


def test_roles_empty_role_name_rejected():
    body = {
        "manuscripts": [
            {"id": "m1", "role_requirements": {"": 1}},
        ],
        "reviewers": [],
    }
    resp = client.post("/assign-roles", json=body)
    assert resp.status_code == 422


def test_roles_extra_field_rejected():
    body = {
        "manuscripts": [],
        "reviewers": [],
        "required_reviewers": 2,
    }
    resp = client.post("/assign-roles", json=body)
    assert resp.status_code == 422


# ---------- 与 /assign 并存 ----------

def test_assign_endpoint_still_works():
    """新增 /assign-roles 不影响既有 /assign。"""
    body = {
        "required_reviewers": 1,
        "manuscripts": [
            {"id": "m1", "authors": [], "topics": ["t"]},
        ],
        "reviewers": [
            {"id": "r1", "topics": ["t"], "capacity": 1},
        ],
    }
    data = client.post("/assign", json=body).json()
    assert data["status"] == "feasible"
    assert data["assignments"] == [{"manuscript_id": "m1", "reviewer_ids": ["r1"]}]


# ---------- 暴力枚举对拍 ----------

def test_roles_matches_brute_force():
    """小实例暴力枚举：可行性判定与暴力枚举一致，可行解满足全部约束。"""
    rng = random.Random(20260926)
    roles_pool = ["chair", "reviewer", "meta"]
    topics_pool = ["t1", "t2"]
    authors_pool = ["a0", "a1", "a2"]
    checked = {"feasible": 0, "infeasible": 0}
    for _ in range(120):
        manuscripts = [
            {
                "id": f"m{i}",
                "authors": rng.sample(authors_pool, rng.randint(0, 1)),
                "topics": rng.sample(topics_pool, rng.randint(0, 2)),
                "role_requirements": {
                    role: rng.randint(0, 2)
                    for role in rng.sample(roles_pool, rng.randint(1, 2))
                },
                **({"track": rng.choice(["A", "B"])} if rng.random() < 0.4 else {}),
            }
            for i in range(rng.randint(1, 3))
        ]
        reviewers = [
            {
                "id": f"r{j}",
                "topics": rng.sample(topics_pool, rng.randint(0, 2)),
                "capacity": rng.randint(0, 2),
                "roles": rng.sample(roles_pool, rng.randint(0, 2)),
                "conflicted_authors": rng.sample(authors_pool, rng.randint(0, 1)),
                **(
                    {"track_capacities": {
                        t: rng.randint(0, 1) for t in rng.sample(["A", "B"], 1)
                    }}
                    if rng.random() < 0.4 else {}
                ),
            }
            for j in range(rng.randint(1, 4))
        ]
        body = {"manuscripts": manuscripts, "reviewers": reviewers}
        data = client.post("/assign-roles", json=body).json()

        def eligible(m: dict, r: dict) -> bool:
            if r["id"] in m["authors"]:
                return False
            if set(m["authors"]) & set(r.get("conflicted_authors", [])):
                return False
            if m["id"] in r.get("conflicted_manuscripts", []):
                return False
            if m["topics"] and not set(m["topics"]) & set(r["topics"]):
                return False
            return True

        # 暴力枚举：每个 (稿件, 角色) 选指定数量的不同评审人，
        # 同一稿件跨角色不重复，全体满足总容量与赛道限额
        per_m_options = []
        for m in manuscripts:
            per_role = []
            for role, n in sorted(m["role_requirements"].items()):
                cands = [
                    r for r in reviewers
                    if eligible(m, r) and role in r.get("roles", [])
                ]
                per_role.append(list(combinations(cands, n)))
            options = []
            for combo in product(*per_role):
                flat = [r["id"] for picks in combo for r in picks]
                if len(set(flat)) == len(flat):
                    options.append(combo)
            per_m_options.append(options)

        feasible = False
        for choice in product(*per_m_options):
            loads = {r["id"]: 0 for r in reviewers}
            track_loads: dict[str, dict[str, int]] = {
                r["id"]: {} for r in reviewers
            }
            for m, combo in zip(manuscripts, choice):
                for picks in combo:
                    for r in picks:
                        loads[r["id"]] += 1
                        track = m.get("track")
                        if track is not None:
                            per = track_loads[r["id"]]
                            per[track] = per.get(track, 0) + 1
            ok = all(loads[r["id"]] <= r["capacity"] for r in reviewers)
            for r in reviewers:
                caps = r.get("track_capacities") or {}
                for track, n in track_loads[r["id"]].items():
                    if track in caps and n > caps[track]:
                        ok = False
            if ok:
                feasible = True
                break

        if feasible:
            assert data["status"] == "feasible"
            _verify_role_constraints(body, data)
            checked["feasible"] += 1
        else:
            assert data["status"] == "infeasible"
            assert data["assignments"] is None
            assert sum(u["missing"] for u in data["unsatisfied"]) > 0
            checked["infeasible"] += 1
    assert checked["feasible"] > 0
    assert checked["infeasible"] > 0
