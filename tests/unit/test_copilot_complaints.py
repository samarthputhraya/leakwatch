from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from leakwatch.common.config import load_config
from leakwatch.copilot import testing
from leakwatch.copilot.complaints import (
    extract_complaint,
    extract_rules,
    match_complaint,
    parse_onset,
    process_complaint,
)
from leakwatch.copilot.llm import ScriptedLLM, reply
from leakwatch.copilot.tools import ToolContext

CFG = load_config(environ={})["copilot"]["complaints"]
REPORTED = pd.Timestamp("2019-03-10 09:00", tz="UTC")  # a Sunday


@pytest.mark.parametrize(
    ("text", "area", "issue"),
    [
        ("No pressure on our street in area C since this morning.", "C", "no_water"),
        ("Water is gushing out of the road near zone A.", "A", "visible_leak"),
        ("Very low pressure at our tap in area B since yesterday.", "B", "low_pressure"),
        ("There is a hissing noise from the pipe outside our house in Area a.", "A", "noise"),
        ("Brown, dirty water from the kitchen tap since 8 am, area C.", "C", "water_quality"),
        ("Our taps have been dry for three hours.", None, "no_water"),
        ("Water pooling on the street for the last hour.", None, "visible_leak"),
        ("The bill is wrong.", None, "other"),
    ],
)
def test_rule_extraction(text, area, issue):
    ex = extract_rules(text, REPORTED, CFG)
    assert (ex.area, ex.issue_type, ex.method) == (area, issue, "rules")
    assert ex.time_reported == REPORTED.to_pydatetime()


@pytest.mark.parametrize(
    ("text", "onset"),
    [
        ("since this morning", "2019-03-10 06:00"),
        ("since last night", "2019-03-09 22:00"),
        ("since yesterday", "2019-03-09 00:00"),
        ("since yesterday evening", "2019-03-09 18:00"),
        ("for the last 3 hours", "2019-03-10 06:00"),
        ("for the last hour", "2019-03-10 08:00"),
        ("two hours ago", "2019-03-10 07:00"),
        ("since 8 am", "2019-03-10 08:00"),
        ("since 6:30", "2019-03-10 06:30"),
        ("since 10 pm", "2019-03-09 22:00"),
        ("since Friday", "2019-03-08 00:00"),
        ("since today", "2019-03-10 00:00"),
    ],
)
def test_onset_phrases(text, onset):
    got, phrase = parse_onset(text, REPORTED, CFG)
    assert got == pd.Timestamp(onset, tz="UTC") and phrase


def test_no_onset_phrase():
    assert parse_onset("the water is brown", REPORTED, CFG) == (None, None)


def test_node_mentions_use_the_network_for_the_area():
    ex = extract_rules("Water pooling near node n7.", REPORTED, CFG, network=testing.toy_network)
    assert ex.area == "C"


def alerts_frame() -> pd.DataFrame:
    rows = [
        r
        for r in testing.alert_rows()
        if not (r["alert_id"] == testing.ALERT_A1 and r["version"] == 1)
    ]
    return pd.DataFrame(rows)[["alert_id", "status", "area", "start_ts", "end_ts"]]


def test_match_statuses():
    alerts = alerts_frame()
    ex = extract_rules("No water in area A since this morning", REPORTED, CFG)
    m = match_complaint(ex, alerts, 24)
    assert m["status"] == "matched" and m["alert_id"] == testing.ALERT_A1
    later = extract_rules("No water in area A", pd.Timestamp("2019-03-14 12:00", tz="UTC"), CFG)
    assert match_complaint(later, alerts, 24)["status"] == "no_matching_alert"
    # the open area-C alert stays active after its latest window
    open_c = extract_rules(
        "Low pressure in area C", pd.Timestamp("2019-03-20 12:00", tz="UTC"), CFG
    )
    assert match_complaint(open_c, alerts, 24)["alert_id"] == testing.ALERT_C2
    no_area = extract_rules("No water", REPORTED, CFG)
    assert match_complaint(no_area, alerts, 24)["status"] == "need_area"
    empty = match_complaint(ex, alerts.iloc[0:0], 24)
    assert empty["status"] == "no_matching_alert"


def test_llm_assisted_extraction_and_fallback():
    good = ScriptedLLM(
        [reply('{"area": "B", "issue_type": "low_pressure", "onset_text": "since 7 am"}')]
    )
    ex = extract_complaint("weak water near the school", REPORTED, CFG, llm=good)
    assert (ex.area, ex.issue_type, ex.method) == ("B", "low_pressure", "llm")
    assert ex.onset == pd.Timestamp("2019-03-10 07:00", tz="UTC").to_pydatetime()
    assert good.requests[0]["json_mode"] is True
    bad = ScriptedLLM([reply('{"area": "Z"}')])
    ex = extract_complaint("No water in area C", REPORTED, CFG, llm=bad)
    assert (ex.area, ex.method) == ("C", "rules")


def test_process_complaint_on_the_tiny_lake(tmp_path):
    root = testing.build_tiny_lake(tmp_path / "lake")
    cfg = copy.deepcopy(load_config(environ={}))
    ctx = ToolContext(full_cfg=cfg, lake_root=root, network=testing.toy_network())
    out = process_complaint(
        ctx, "Water gushing from the road in zone C", testing.ts("2019-03-12 12:00")
    )
    assert out["match"]["status"] == "matched" and out["match"]["alert_id"] == testing.ALERT_C2
    assert "synthetic" in out["synthetic_note"] and "already working" in out["reply"]
    json.dumps(out)
    assert ctx.complaints == [out]
    default_time = process_complaint(ctx, "No water")
    assert default_time["extracted"]["time_reported"] == "2019-03-12 14:00 UTC"  # the data clock
    assert default_time["match"]["status"] == "need_area"
