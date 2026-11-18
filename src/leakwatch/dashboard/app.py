"""LeakWatch control-room dashboard (PRD Phase 7): ``make dashboard`` (Streamlit).

Two modes:

- **Live demo** reads the demo lake written by ``demo.runner`` (``data/demo/lake``) and its state
  file, refreshes every few seconds, and sends ``demo.commands`` (inject a leak at a clicked
  junction, change speed, reset);
- **History** reads the main lake (BattLeDIM 2018–2019 batch alerts).

Panels: L-Town map (pipes, sensors, alert area, localizer top 5, the injected leak after
"reveal"), alert feed, alert details with the work order, sensor charts, and the copilot.
Every number on screen is read from the lake (DuckDB) or the demo state file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from leakwatch.common import lake, paths
from leakwatch.common.config import load_config

AREA_COLORS = {"A": "#4C78A8", "B": "#F58518", "C": "#54A24B"}
st.set_page_config(page_title="LeakWatch control room", page_icon="💧", layout="wide")


# ------------------------------------------------------------------------------- data access


@st.cache_resource
def network_layout() -> dict[str, Any]:
    """Node coordinates, pipe segments, areas and sensors of L-Town (loaded once)."""
    from leakwatch.data import network

    wn = network.load_network()
    coords = network.node_coordinates(wn)
    areas = network.node_areas(wn, load_config()["network"]["area_anchors"])
    xs, ys = [], []
    for _, link in wn.links():
        a, b = coords.loc[link.start_node_name], coords.loc[link.end_node_name]
        xs += [a.x, b.x, None]
        ys += [a.y, b.y, None]
    sensors = network.parse_sensor_config()
    return {
        "coords": coords,
        "areas": areas,
        "pipe_x": xs,
        "pipe_y": ys,
        "pressure": sensors["pressure"],
        "junctions": sorted(wn.junction_name_list),
    }


def demo_lake() -> Path:
    """Root of the demo lake."""
    from leakwatch.demo.runner import demo_lake as _demo_lake

    return _demo_lake()


def demo_state() -> dict[str, Any]:
    """The demo runner's state file."""
    from leakwatch.demo.runner import read_state

    return read_state()


def query(root: Path | None, sql: str) -> pd.DataFrame:
    """Run SQL on the lake at ``root`` (empty frame if a view does not exist yet)."""
    try:
        con = lake.connect(lake_root=root) if root else lake.connect()
        return lake.query_df(con, sql)
    except Exception:
        return pd.DataFrame()


def alerts(root: Path | None, limit: int = 50) -> pd.DataFrame:
    """Current alerts, newest first."""
    return query(
        root,
        f"""SELECT alert_id, origin, status, start_ts, end_ts, area, kmeans_score_max, threshold,
                   est_loss_m3h, windows_above, top_sensors, gnn_top5, work_order_json,
                   work_order_generator
            FROM alerts ORDER BY start_ts DESC LIMIT {int(limit)}""",
    )


def series_chart(
    root: Path | None, table: str, series_id: str, around: pd.Timestamp | None
) -> go.Figure | None:
    """Window means and residuals of one series (feature table) around a time."""
    where = ""
    if around is not None:
        lo, hi = around - pd.Timedelta(hours=12), around + pd.Timedelta(hours=6)
        where = f"AND window_end BETWEEN '{lo}' AND '{hi}'"
    df = query(
        root,
        f"""SELECT window_end, "{series_id}__mean" AS mean, "{series_id}__residual" AS residual
            FROM {table} WHERE 1=1 {where} ORDER BY window_end""",
    )
    if df.empty:
        return None
    fig = go.Figure()
    fig.add_scatter(x=df["window_end"], y=df["mean"], name="1 h mean", mode="lines")
    fig.add_scatter(
        x=df["window_end"], y=df["residual"], name="residual vs baseline", mode="lines", yaxis="y2"
    )
    if around is not None:
        fig.add_vline(x=around, line_dash="dash", line_color="red")
    fig.update_layout(
        height=280,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title=series_id,
        yaxis={"title": "value"},
        yaxis2={"title": "residual", "overlaying": "y", "side": "right"},
        legend={"orientation": "h"},
    )
    return fig


# --------------------------------------------------------------------------------------- map


def network_map(alert: pd.Series | None, leaks: list[dict[str, Any]], reveal: bool) -> go.Figure:
    """L-Town: pipes, junctions coloured by area (alert area red), sensors, top 5, true leak."""
    net = network_layout()
    coords, areas = net["coords"], net["areas"]
    junctions = coords.loc[net["junctions"]]
    fig = go.Figure()
    fig.add_scattergl(
        x=net["pipe_x"],
        y=net["pipe_y"],
        mode="lines",
        line={"color": "#bbbbbb", "width": 1},
        hoverinfo="skip",
        name="pipes",
    )
    alert_area = None if alert is None else alert.get("area")
    colors = [
        ("#d62728" if areas.get(n) == alert_area else AREA_COLORS.get(areas.get(n), "#999"))
        for n in junctions.index
    ]
    fig.add_scattergl(
        x=junctions["x"],
        y=junctions["y"],
        mode="markers",
        marker={"size": 5, "color": colors, "opacity": 0.75},
        customdata=junctions.index,
        hovertemplate="junction %{customdata}<extra></extra>",
        name="junctions (click to choose a leak site)",
    )
    s = coords.loc[net["pressure"]]
    fig.add_scattergl(
        x=s["x"],
        y=s["y"],
        mode="markers",
        marker={"size": 9, "symbol": "triangle-up", "color": "#1f1f1f"},
        customdata=s.index,
        hovertemplate="pressure sensor %{customdata}<extra></extra>",
        name="pressure sensors",
    )
    if (
        alert is not None
        and isinstance(alert.get("gnn_top5"), list | tuple)
        and len(alert["gnn_top5"])
    ):
        cands = [c["node_id"] for c in alert["gnn_top5"] if c["node_id"] in coords.index]
        c = coords.loc[cands]
        fig.add_scattergl(
            x=c["x"],
            y=c["y"],
            mode="markers+text",
            text=[f"#{i + 1}" for i in range(len(cands))],
            textposition="top center",
            marker={
                "size": 16,
                "symbol": "star",
                "color": "#ff7f0e",
                "line": {"width": 1, "color": "black"},
            },
            customdata=c.index,
            hovertemplate="localizer candidate %{customdata}<extra></extra>",
            name="localizer top 5",
        )
    if reveal and leaks:
        nodes = [lk["node_id"] for lk in leaks if lk["node_id"] in coords.index]
        c = coords.loc[nodes]
        fig.add_scattergl(
            x=c["x"],
            y=c["y"],
            mode="markers",
            marker={"size": 18, "symbol": "x", "color": "#d62728"},
            customdata=c.index,
            hovertemplate="true leak %{customdata}<extra></extra>",
            name="true leak",
        )
    fig.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"orientation": "h", "y": -0.02},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        dragmode="pan",
    )
    return fig


# ----------------------------------------------------------------------------------- panels


def alert_details(row: pd.Series, root: Path | None, table: str) -> None:
    """Explanation, localizer candidates, work order and the top sensor's chart."""
    st.markdown(
        f"**Alert** `{row['alert_id'][:8]}` · status **{row['status']}** · area **{row['area']}**"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("score / threshold", f"{row['kmeans_score_max']:.1f} / {row['threshold']:.1f}")
    loss = row.get("est_loss_m3h")
    c2.metric(
        "estimated loss",
        "n/a" if pd.isna(loss) else f"{loss:.1f} m³/h",
        help="Excess inflow vs the hour-of-week baseline (estimate)",
    )
    c3.metric("windows above τ", int(row["windows_above"]))
    st.caption(f"Fired at {pd.Timestamp(row['start_ts']):%Y-%m-%d %H:%M} UTC (simulated time)")
    tops = row.get("top_sensors")
    if isinstance(tops, list | tuple) and len(tops):
        st.write(
            "Sensors that moved most: "
            + ", ".join(f"{t['sensor_id']} ({t['contribution']:.0%})" for t in tops)
        )
    gnn = row.get("gnn_top5")
    if isinstance(gnn, list | tuple) and len(gnn):
        st.write("Localizer top 5: " + ", ".join(f"{c['node_id']} ({c['prob']:.0%})" for c in gnn))
    wo = row.get("work_order_json")
    if isinstance(wo, str) and wo:
        order = json.loads(wo)
        label = "template fallback" if row.get("work_order_generator") == "template" else "LLM"
        with st.expander(f"Work order · {order.get('priority')} · {label}", expanded=True):
            st.write(order.get("summary"))
            st.write(
                "**Pipes to inspect, in order:** " + ", ".join(order.get("pipes_to_inspect", []))
            )
            if order.get("estimated_loss_m3_per_hour") is not None:
                st.write(f"**Estimated loss:** {order['estimated_loss_m3_per_hour']} m³/h")
            for action in order.get("recommended_actions", []):
                st.write(f"- {action}")
            cites = order.get("manual_citations", [])
            if cites:
                st.caption(
                    "Sources: "
                    + "; ".join(f"{c['doc']}, {c['chapter']}, p. {c['page']}" for c in cites)
                )
    else:
        st.info("Work order is being prepared…")
    if st.button(
        "How much water have we lost since this alert started?", key=f"loss-{row['alert_id']}"
    ):
        from leakwatch.copilot.tools import ToolContext, estimate_loss

        res = estimate_loss(ToolContext(lake_root=root), {"alert_id": row["alert_id"]})
        st.json(res)
    if isinstance(tops, list | tuple) and len(tops):
        fig = series_chart(root, table, tops[0]["sensor_id"], pd.Timestamp(row["start_ts"]))
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)


def copilot_panel(root: Path | None) -> None:
    """Chat with the copilot (needs LLM_* in .env; tool shortcuts work without)."""
    from leakwatch.copilot.settings import llm_settings

    st.subheader("Copilot")
    settings = llm_settings()
    if not settings.configured:
        st.caption(
            "No language model configured (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env): "
            "chat is off; alert details and work orders (template) still work."
        )
    history = st.session_state.setdefault("chat", [])
    for msg in history:
        st.chat_message(msg["role"]).write(msg["content"])
    question = st.chat_input(
        "Ask about alerts, sensors or procedures", disabled=not settings.configured
    )
    if question:
        from leakwatch.copilot.agent import CopilotAgent
        from leakwatch.copilot.llm import build_llm
        from leakwatch.copilot.tools import ToolContext

        ctx = ToolContext(lake_root=root)
        agent = CopilotAgent(build_llm(settings, ctx.cfg["llm"]), ctx)
        st.chat_message("user").write(question)
        answer = agent.ask(question, history)
        st.chat_message("assistant").write(answer.answer)
        for chart in answer.charts or []:
            data = pd.DataFrame(chart.get("data") or chart.get("points") or [])
            if not data.empty:
                st.line_chart(data.set_index(data.columns[0]))
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer.answer},
        ]


# ---------------------------------------------------------------------------------- layout


def demo_controls(state: dict[str, Any], selected_node: str | None) -> None:
    """Inject a leak, change speed, reset (sent on ``demo.commands``)."""
    from leakwatch.demo.runner import send_command

    st.subheader("Demo controls")
    net = network_layout()
    node = st.selectbox(
        "Leak site (junction)",
        net["junctions"],
        index=net["junctions"].index(selected_node) if selected_node in net["junctions"] else 0,
    )
    diameter_mm = st.slider(
        "Leak size (orifice diameter, mm)",
        10,
        50,
        int(load_config()["demo"]["leak_diameter_m"] * 1000),
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("Inject leak", type="primary"):
        send_command("inject_leak", node_id=node, diameter_m=diameter_mm / 1000)
        st.toast(f"Leak injected at {node}")
    speed = c2.selectbox(
        "Speed",
        [180, 360, 720, 1440],
        index=[180, 360, 720, 1440].index(int(state.get("speed", 360)))
        if int(state.get("speed", 360)) in [180, 360, 720, 1440]
        else 1,
    )
    if c2.button("Set speed"):
        send_command("set_speed", speed=speed)
    if c3.button("Reset"):
        send_command("reset")
        st.session_state.pop("alert_pick", None)


def main() -> None:
    """Render the page."""
    st.title("LeakWatch · L-Town control room")
    mode = st.sidebar.radio("Mode", ["Live demo", "History (BattLeDIM 2018–2019)"])
    live = mode == "Live demo"
    root = demo_lake() if live else None
    table = "features_stream" if live else "features"
    st.sidebar.toggle("Reveal the true leak", key="reveal")
    st.sidebar.caption(
        "L-Town is the BattLeDIM benchmark network, not a real city. Times are simulated."
    )

    @st.fragment(run_every="3s" if live else None)
    def live_view() -> None:
        state = demo_state() if live else {}
        df = alerts(root)
        open_alerts = df[df["status"] == "open"] if not df.empty else df
        if live:
            sim = state.get("sim_ts")
            status = state.get("status", "not running")
            line = (
                f"Status: **{status}** · simulated time {pd.Timestamp(sim):%Y-%m-%d %H:%M} UTC"
                f" · speed {state.get('speed')}×"
                if sim
                else f"Status: **{status}** (start it with `make demo-runner`)"
            )
            st.markdown(line)
        if df.empty or open_alerts.empty:
            st.success("All normal — no open alerts")
        else:
            st.error(
                f"{len(open_alerts)} open alert(s) — area "
                + ", ".join(sorted(open_alerts["area"].dropna().unique()))
            )
        left, right = st.columns([3, 2])
        selected_id = st.session_state.get("alert_pick")
        row = None
        if not df.empty:
            ids = df["alert_id"].tolist()
            if selected_id not in ids:
                st.session_state.pop("alert_pick", None)
                selected_id = ids[0]
            row = df.set_index("alert_id").loc[selected_id].copy()
            row["alert_id"] = selected_id
        with left:
            event = st.plotly_chart(
                network_map(row, state.get("leaks", []), st.session_state.get("reveal", False)),
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="map",
            )
            points = (
                (event or {}).get("selection", {}).get("points", [])
                if isinstance(event, dict)
                else getattr(getattr(event, "selection", None), "points", [])
            )
            if points:
                st.session_state["selected_node"] = points[0].get("customdata")
            clicked = st.session_state.get("selected_node")
            if live and clicked in network_layout()["junctions"]:
                from leakwatch.demo.runner import send_command

                size_m = float(load_config()["demo"]["leak_diameter_m"])
                if st.button(
                    f"Inject a {size_m * 1000:.0f} mm leak at junction {clicked}",
                    type="primary",
                    key="inject-clicked",
                ):
                    send_command("inject_leak", node_id=clicked, diameter_m=size_m)
                    st.toast(f"Leak injected at {clicked}")
            elif live:
                st.caption("Click a junction on the map to choose a leak site.")
        with right:
            st.subheader("Alerts")
            if df.empty:
                st.caption("No alerts yet.")
            else:
                view = df[["alert_id", "status", "start_ts", "area", "est_loss_m3h"]].copy()
                view["alert_id"] = view["alert_id"].str[:8]
                st.dataframe(view.head(10), hide_index=True, use_container_width=True)
                st.selectbox(
                    "Alert",
                    ids,
                    index=ids.index(selected_id),
                    format_func=lambda a: a[:8],
                    key="alert_pick",
                )
                alert_details(row, root, table)
        if live and state.get("leaks") and st.session_state.get("reveal"):
            st.caption(
                "Injected leaks: "
                + "; ".join(
                    f"{lk['node_id']} ({lk['diameter_m'] * 1000:.0f} mm, "
                    f"~{lk.get('est_flow_m3h', '?')} m³/h) at {lk['start_ts']}"
                    for lk in state["leaks"]
                )
            )

    live_view()
    if live:
        with st.sidebar:
            demo_controls(demo_state(), st.session_state.get("selected_node"))
    copilot_panel(root)
    st.sidebar.caption(
        f"Page rendered {datetime.now(UTC):%H:%M:%S} UTC · data dir {paths.data_dir()}"
    )


main()
