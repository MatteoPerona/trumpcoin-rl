import streamlit as st
import json
import os
import glob
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── App config ──
st.set_page_config(page_title="TRUMP Coin RL Dashboard", layout="wide")
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

ACTION_NAMES = ["Heavy Sell", "Light Sell", "Hold", "Light Buy", "Heavy Buy"]


def save_run(model_name, result):
    """Persist a run result to JSON + model artifact."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ds = result.get('env_config', {}).get('dataset', 'trump')
    run_id = f"{model_name}_{ds}_{ts}"

    # Save model artifact separately (not in JSON)
    if 'q_table' in result:
        np.save(os.path.join(RUNS_DIR, f"{run_id}.npy"), result.pop('q_table'))
    if 'model_state' in result:
        import torch
        torch.save(result.pop('model_state'), os.path.join(RUNS_DIR, f"{run_id}.pt"))

    payload = {"run_id": run_id, "model": model_name, "timestamp": ts, **result}
    path = os.path.join(RUNS_DIR, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return run_id


def load_all_runs():
    """Load every JSON run file into a list of dicts."""
    runs = []
    for fp in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        with open(fp) as f:
            runs.append(json.load(f))
    return runs


# ── Sidebar navigation ──
page = st.sidebar.radio("Navigation", [
    "🚀 Run Experiment",
    "📊 Explore Past Runs",
    "🔍 Inspect Models",
])

# ======================================================================
# PAGE 1 – RUN EXPERIMENT
# ======================================================================
if page == "🚀 Run Experiment":
    st.title("🚀 Run Experiment")

    ALL_MODELS = [
        "Tabular Q-Learning",
        "Double DQN",
        "REINFORCE + Baseline",
        "Buy & Hold",
        "SMA Crossover",
        "Random Agent",
    ]
    model = st.radio("Model", ALL_MODELS, horizontal=True)
    is_baseline = model in ("Buy & Hold", "SMA Crossover", "Random Agent")

    st.markdown("---")
    col_env, col_model = st.columns(2)

    with col_env:
        st.subheader("Environment")
        dataset = st.radio("Dataset", ["TRUMP", "BTC"], horizontal=True,
                           help="Which asset to evaluate on")
        risk_aversion = st.slider("Risk Aversion (λ)", 0.0, 5.0, 0.5, 0.1,
                                  help="Quadratic penalty on returns. Higher → more conservative.")
        tx_cost = st.number_input("Transaction Cost", 0.0000, 0.0100, 0.0010, 0.0001, format="%.4f")
        train_ratio = st.slider("Train / Test Split", 0.50, 0.90, 0.70, 0.05)

    with col_model:
        if model == "Tabular Q-Learning":
            st.subheader("Q-Learning Hyperparameters")
            alpha = st.slider("Learning Rate (α)", 0.01, 1.0, 0.10, 0.01)
            gamma = st.slider("Discount Factor (γ)", 0.80, 1.0, 0.99, 0.01)
            eps_decay = st.slider("Epsilon Decay", 0.50, 0.999, 0.80, 0.001)
            episodes = st.number_input("Episodes", 5, 200, 20)
        elif model == "Double DQN":
            st.subheader("DQN Hyperparameters")
            lr = st.select_slider("Learning Rate", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3], value=1e-4)
            gamma = st.slider("Discount Factor (γ)", 0.80, 1.0, 0.99, 0.01)
            hidden_dim = st.select_slider("Hidden Dim", [32, 64, 128, 256], value=64)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.1, 0.05)
            weight_decay = st.select_slider("Weight Decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3], value=1e-5)
            state_noise = st.slider("State Noise (σ)", 0.0, 0.10, 0.01, 0.005)
            eps_decay = st.slider("Epsilon Decay", 0.80, 0.999, 0.95, 0.001)
            episodes = st.number_input("Episodes", 5, 200, 50)
        elif model == "REINFORCE + Baseline":
            st.subheader("REINFORCE Hyperparameters")
            policy_lr = st.select_slider("Policy Learning Rate", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3], value=1e-4)
            value_lr = st.select_slider("Value Learning Rate", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3], value=1e-3)
            gamma = st.slider("Discount Factor (γ)", 0.80, 1.0, 0.99, 0.01)
            hidden_dim = st.select_slider("Hidden Dim", [32, 64, 128, 256], value=64)
            episode_length = st.number_input("Episode Window (hours)", 24, 1000, 168,
                                             help="Each training episode samples a random contiguous window from the train split.")
            entropy_coef = st.slider("Entropy Coef", 0.0, 0.05, 0.0, 0.005,
                                     help="Small positive values encourage exploration.")
            episodes = st.number_input("Episodes", 5, 200, 50)
        elif model == "SMA Crossover":
            st.subheader("SMA Parameters")
            sma_short = st.number_input("Short Window (hours)", 4, 100, 12)
            sma_long = st.number_input("Long Window (hours)", 12, 500, 48)
            st.caption("Evaluates on the test set only — no training phase.")
        elif model == "Buy & Hold":
            st.subheader("Buy & Hold")
            st.info("Buys 100% on step 0, then holds. No parameters to configure.")
            st.caption("Evaluates on the test set only — no training phase.")
        elif model == "Random Agent":
            st.subheader("Random Agent")
            st.info("Takes a random action every step. No parameters to configure.")
            st.caption("Evaluates on the test set only — no training phase.")

    btn_label = "▶️  Run Evaluation" if is_baseline else "▶️  Start Training"
    st.markdown("---")
    if st.button(btn_label, type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_area = st.empty()

        def on_progress(ep, total, m):
            progress_bar.progress(ep / total)
            parts = [
                f"Episode {ep}/{total}",
                f"Reward: {m['reward']:.2f}",
                f"Portfolio: ${m['portfolio']:.2f}",
            ]
            if 'epsilon' in m:
                parts.append(f"ε: {m['epsilon']:.3f}")
            if 'policy_loss' in m:
                parts.append(f"Policy Loss: {m['policy_loss']:.4f}")
            if 'value_loss' in m:
                parts.append(f"Value Loss: {m['value_loss']:.4f}")
            status_area.text(" | ".join(parts))

        with st.spinner("Running…"):
            if model == "Tabular Q-Learning":
                from td_agent import run_experiment
                config = dict(alpha=alpha, gamma=gamma, epsilon_decay=eps_decay,
                              episodes=int(episodes), risk_aversion=risk_aversion,
                              tx_cost=tx_cost, train_ratio=train_ratio,
                              dataset=dataset.lower())
                result = run_experiment(config, progress_cb=on_progress)
            elif model == "Double DQN":
                from dqn_agent import run_experiment
                config = dict(lr=lr, gamma=gamma, hidden_dim=hidden_dim,
                              dropout_rate=dropout_rate, weight_decay=weight_decay,
                              state_noise=state_noise, epsilon_decay=eps_decay,
                              episodes=int(episodes), risk_aversion=risk_aversion,
                              tx_cost=tx_cost, train_ratio=train_ratio,
                              dataset=dataset.lower())
                result = run_experiment(config, progress_cb=on_progress)
            elif model == "REINFORCE + Baseline":
                from reinforce_agent import run_experiment
                config = dict(policy_lr=policy_lr, value_lr=value_lr, gamma=gamma,
                              hidden_dim=hidden_dim, episode_length=int(episode_length),
                              entropy_coef=entropy_coef,
                              episodes=int(episodes), risk_aversion=risk_aversion,
                              tx_cost=tx_cost, train_ratio=train_ratio,
                              dataset=dataset.lower())
                result = run_experiment(config, progress_cb=on_progress)
            else:
                from baselines import run_experiment
                strategy_map = {"Buy & Hold": "buy_hold",
                                "SMA Crossover": "sma_crossover",
                                "Random Agent": "random"}
                config = dict(strategy=strategy_map[model],
                              risk_aversion=risk_aversion, tx_cost=tx_cost,
                              train_ratio=train_ratio, dataset=dataset.lower())
                if model == "SMA Crossover":
                    config['sma_short'] = int(sma_short)
                    config['sma_long'] = int(sma_long)
                result = run_experiment(config)

        progress_bar.progress(1.0)
        status_area.success("✅ Complete!")

        model_key_map = {"Tabular Q-Learning": "q_learning", "Double DQN": "dqn",
                         "REINFORCE + Baseline": "reinforce",
                         "Buy & Hold": "buy_hold", "SMA Crossover": "sma_crossover",
                         "Random Agent": "random"}
        run_id = save_run(model_key_map[model], result)
        st.info(f"Run saved as `{run_id}`")

        ev = result['evaluation']
        c1, c2, c3 = st.columns(3)
        c1.metric("Test Portfolio", f"${ev['final_portfolio']:.2f}",
                  f"{((ev['final_portfolio'] / 10000) - 1) * 100:+.1f}%")
        c2.metric("Test Reward", f"{ev['total_reward']:.4f}")
        total_actions = sum(ev['action_distribution'].values())
        dominant = max(ev['action_distribution'], key=ev['action_distribution'].get)
        c3.metric("Dominant Action", dominant,
                  f"{ev['action_distribution'][dominant] / max(total_actions, 1) * 100:.0f}%")

        st.subheader("Training Reward Curve")
        fig = px.line(x=list(range(1, len(result['training']['rewards']) + 1)),
                      y=result['training']['rewards'],
                      labels={'x': 'Episode', 'y': 'Reward'})
        fig.update_layout(height=350, margin=dict(t=10, b=30))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Test Action Distribution")
            fig2 = px.bar(x=list(ev['action_distribution'].keys()),
                          y=list(ev['action_distribution'].values()),
                          labels={'x': 'Action', 'y': 'Count'})
            fig2.update_layout(height=300, margin=dict(t=10, b=30))
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            st.subheader("Test Portfolio Curve")
            fig3 = px.line(y=ev['portfolio_curve'],
                           labels={'x': 'Step', 'y': 'Portfolio ($)'})
            fig3.update_layout(height=300, margin=dict(t=10, b=30))
            st.plotly_chart(fig3, use_container_width=True)


# ======================================================================
# PAGE 2 – EXPLORE PAST RUNS
# ======================================================================
elif page == "📊 Explore Past Runs":
    st.title("📊 Explore Past Runs")

    runs = load_all_runs()
    if not runs:
        st.info("No runs saved yet. Go to **Run Experiment** to create one.")
        st.stop()

    table_data = []
    for r in runs:
        ev = r.get('evaluation', {})
        table_data.append({
            'Run ID': r['run_id'],
            'Model': r['model'],
            'Dataset': r.get('env_config', {}).get('dataset', 'trump').upper(),
            'Timestamp': r.get('timestamp', ''),
            'Episodes': r.get('hyperparameters', {}).get('episodes', ''),
            'λ': r.get('env_config', {}).get('risk_aversion', ''),
            'Test Portfolio': f"${ev.get('final_portfolio', 0):.2f}",
            'Test Reward': f"{ev.get('total_reward', 0):.4f}",
        })

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    run_ids = [r['run_id'] for r in runs]
    selected = st.multiselect("Select runs to compare", run_ids,
                              default=run_ids[-min(2, len(run_ids)):])

    if not selected:
        st.stop()

    selected_runs = [r for r in runs if r['run_id'] in selected]

    st.subheader("Training Reward Curves")
    fig = go.Figure()
    for r in selected_runs:
        rewards = r.get('training', {}).get('rewards', [])
        fig.add_trace(go.Scatter(
            x=list(range(1, len(rewards) + 1)), y=rewards,
            mode='lines', name=r['run_id']
        ))
    fig.update_layout(xaxis_title="Episode", yaxis_title="Reward",
                      height=400, margin=dict(t=10, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Evaluation Comparison")
    cols = st.columns(len(selected_runs))
    for col, r in zip(cols, selected_runs):
        ev = r.get('evaluation', {})
        with col:
            st.markdown(f"**{r['run_id']}**")
            st.metric("Portfolio", f"${ev.get('final_portfolio', 0):.2f}")
            st.metric("Reward", f"{ev.get('total_reward', 0):.4f}")
            ad = ev.get('action_distribution', {})
            if ad:
                fig_a = px.bar(x=list(ad.keys()), y=list(ad.values()),
                               labels={'x': '', 'y': 'Count'})
                fig_a.update_layout(height=250, margin=dict(t=10, b=10),
                                    showlegend=False)
                st.plotly_chart(fig_a, use_container_width=True)

    st.subheader("Run Details")
    for r in selected_runs:
        with st.expander(r['run_id']):
            c1, c2 = st.columns(2)
            with c1:
                st.json(r.get('hyperparameters', {}))
            with c2:
                st.json(r.get('env_config', {}))
            curve = r.get('evaluation', {}).get('portfolio_curve', [])
            if curve:
                fig_p = px.line(y=curve, labels={'x': 'Step', 'y': 'Portfolio ($)'})
                fig_p.update_layout(height=250, margin=dict(t=10, b=10))
                st.plotly_chart(fig_p, use_container_width=True)


# ======================================================================
# PAGE 3 – INSPECT MODELS
# ======================================================================
elif page == "🔍 Inspect Models":
    st.title("🔍 Inspect Models")

    runs = load_all_runs()
    if not runs:
        st.info("No runs saved yet. Go to **Run Experiment** to create one.")
        st.stop()

    # Separate runs by model type and check for artifact files
    ql_runs = [r for r in runs
               if r['model'] == 'q_learning'
               and os.path.exists(os.path.join(RUNS_DIR, f"{r['run_id']}.npy"))]
    dqn_runs = [r for r in runs
                if r['model'] == 'dqn'
                and os.path.exists(os.path.join(RUNS_DIR, f"{r['run_id']}.pt"))]
    reinforce_runs = [r for r in runs
                      if r['model'] == 'reinforce'
                      and os.path.exists(os.path.join(RUNS_DIR, f"{r['run_id']}.pt"))]

    if not ql_runs and not dqn_runs and not reinforce_runs:
        st.warning("No runs with saved model artifacts found. "
                   "Run a new experiment to generate inspectable models.")
        st.stop()

    # ── Model type selector ──
    available = []
    if ql_runs:
        available.append("Tabular Q-Learning")
    if dqn_runs:
        available.append("Double DQN")
    if reinforce_runs:
        available.append("REINFORCE + Baseline")

    model_type = st.radio("Model Type", available, horizontal=True)

    # ==================================================================
    # TABULAR Q-LEARNING INSPECTOR
    # ==================================================================
    if model_type == "Tabular Q-Learning":
        run_id = st.selectbox("Select Run", [r['run_id'] for r in ql_runs])
        q_table = np.load(os.path.join(RUNS_DIR, f"{run_id}.npy"))

        st.markdown("---")

        # State dimension labels
        momentum_labels = {0: "📉 Negative", 1: "➡️ Flat", 2: "📈 Positive"}
        volatility_labels = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
        position_labels = {0: "💵 Mostly Cash", 1: "⚖️ Balanced", 2: "🪙 Mostly TRUMP"}
        time_labels = {0: "🌙 Night (UTC)", 1: "☀️ Day (UTC)"}

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Select State")
            momentum = st.selectbox("Momentum (1h Return)",
                                    options=[0, 1, 2],
                                    format_func=lambda x: momentum_labels[x])
            volatility = st.selectbox("Volatility (24h)",
                                      options=[0, 1, 2],
                                      format_func=lambda x: volatility_labels[x])
            position = st.selectbox("Position",
                                    options=[0, 1, 2],
                                    format_func=lambda x: position_labels[x])
            time_bin = st.selectbox("Time of Day",
                                    options=[0, 1],
                                    format_func=lambda x: time_labels[x])

        # Get Q-values for selected state
        q_vals = q_table[momentum, volatility, position, time_bin]
        greedy = int(np.argmax(q_vals))

        with col2:
            st.subheader("Q-Values for Selected State")
            colors = ['#ff6b6b' if i != greedy else '#51cf66' for i in range(5)]
            fig = go.Figure(go.Bar(
                x=ACTION_NAMES, y=q_vals,
                marker_color=colors,
                text=[f"{v:.4f}" for v in q_vals],
                textposition='outside',
            ))
            fig.update_layout(
                height=350, margin=dict(t=10, b=30),
                yaxis_title="Q-Value",
                annotations=[dict(
                    x=ACTION_NAMES[greedy], y=q_vals[greedy],
                    text="★ Greedy", showarrow=True, arrowhead=2,
                    yshift=25, font=dict(size=14, color='#51cf66')
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Full policy table
        st.subheader("Full Policy Table (all 54 states)")
        rows = []
        for m in range(3):
            for v in range(3):
                for p in range(3):
                    for t in range(2):
                        qv = q_table[m, v, p, t]
                        best = int(np.argmax(qv))
                        rows.append({
                            'Momentum': momentum_labels[m],
                            'Volatility': volatility_labels[v],
                            'Position': position_labels[p],
                            'Time': time_labels[t],
                            'Greedy Action': ACTION_NAMES[best],
                            'Q(HeavySell)': f"{qv[0]:.4f}",
                            'Q(LightSell)': f"{qv[1]:.4f}",
                            'Q(Hold)': f"{qv[2]:.4f}",
                            'Q(LightBuy)': f"{qv[3]:.4f}",
                            'Q(HeavyBuy)': f"{qv[4]:.4f}",
                        })
        df_policy = pd.DataFrame(rows)
        st.dataframe(df_policy, use_container_width=True, hide_index=True, height=400)

    # ==================================================================
    # DQN INSPECTOR
    # ==================================================================
    elif model_type == "Double DQN":
        import torch
        from dqn_agent import QNetwork

        run_id = st.selectbox("Select Run", [r['run_id'] for r in dqn_runs])
        run_meta = next(r for r in dqn_runs if r['run_id'] == run_id)
        hp = run_meta.get('hyperparameters', {})

        hidden_dim = int(hp.get('hidden_dim', 64))
        dropout_rate = float(hp.get('dropout_rate', 0.1))

        # Load model
        net = QNetwork(obs_dim=10, action_dim=5,
                       hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        net.load_state_dict(torch.load(
            os.path.join(RUNS_DIR, f"{run_id}.pt"),
            map_location='cpu', weights_only=True
        ))
        net.eval()

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("State Input")
            ret_1h = st.slider("1h Log Return", -0.10, 0.10, 0.0, 0.001, format="%.3f")
            ret_4h = st.slider("4h Log Return", -0.20, 0.20, 0.0, 0.005, format="%.3f")
            ret_24h = st.slider("24h Log Return", -0.50, 0.50, 0.0, 0.01, format="%.2f")
            vol_24h = st.slider("Volatility (24h σ)", 0.0, 0.10, 0.02, 0.005, format="%.3f")
            log_vol = st.slider("Log Volume", 10.0, 30.0, 20.0, 0.5)
            hour_sin = st.slider("Hour Sin", -1.0, 1.0, 0.0, 0.1)
            hour_cos = st.slider("Hour Cos", -1.0, 1.0, 1.0, 0.1)
            day_sin = st.slider("Day Sin", -1.0, 1.0, 0.0, 0.1)
            day_cos = st.slider("Day Cos", -1.0, 1.0, 1.0, 0.1)
            pos = st.slider("Portfolio Position", 0.0, 1.0, 0.0, 0.05)

        state = torch.FloatTensor([
            ret_1h, ret_4h, ret_24h, vol_24h, log_vol,
            hour_sin, hour_cos, day_sin, day_cos, pos
        ]).unsqueeze(0)

        with torch.no_grad():
            q_vals = net(state).squeeze().numpy()

        greedy = int(np.argmax(q_vals))

        with col2:
            st.subheader("Q-Values")
            colors = ['#ff6b6b' if i != greedy else '#51cf66' for i in range(5)]
            fig = go.Figure(go.Bar(
                x=ACTION_NAMES, y=q_vals,
                marker_color=colors,
                text=[f"{v:.4f}" for v in q_vals],
                textposition='outside',
            ))
            fig.update_layout(
                height=350, margin=dict(t=10, b=30),
                yaxis_title="Q-Value",
                annotations=[dict(
                    x=ACTION_NAMES[greedy], y=q_vals[greedy],
                    text="★ Greedy", showarrow=True, arrowhead=2,
                    yshift=25, font=dict(size=14, color='#51cf66')
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

            # Policy surface: sweep position vs return_1h
            st.subheader("Policy Surface (Position × 1h Return)")
            pos_range = np.linspace(0, 1, 21)
            ret_range = np.linspace(-0.05, 0.05, 21)
            grid = np.zeros((len(ret_range), len(pos_range)), dtype=int)

            for i, r_val in enumerate(ret_range):
                for j, p_val in enumerate(pos_range):
                    s = torch.FloatTensor([
                        r_val, ret_4h, ret_24h, vol_24h, log_vol,
                        hour_sin, hour_cos, day_sin, day_cos, p_val
                    ]).unsqueeze(0)
                    with torch.no_grad():
                        grid[i, j] = net(s).argmax().item()

            fig_h = go.Figure(go.Heatmap(
                z=grid,
                x=[f"{p:.2f}" for p in pos_range],
                y=[f"{r:.3f}" for r in ret_range],
                colorscale=[
                    [0.0, '#ff6b6b'], [0.25, '#ffa06b'],
                    [0.5, '#ffe66b'], [0.75, '#6bcfff'], [1.0, '#51cf66']
                ],
                zmin=0, zmax=4,
                colorbar=dict(
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=ACTION_NAMES,
                ),
                hovertemplate="Position: %{x}<br>Return 1h: %{y}<br>"
                              "Action: %{customdata}<extra></extra>",
                customdata=[[ACTION_NAMES[grid[i, j]] for j in range(len(pos_range))]
                            for i in range(len(ret_range))],
            ))
            fig_h.update_layout(
                xaxis_title="Position",
                yaxis_title="1h Log Return",
                height=400, margin=dict(t=10, b=30),
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # ==================================================================
    # REINFORCE INSPECTOR
    # ==================================================================
    elif model_type == "REINFORCE + Baseline":
        import torch
        from reinforce_agent import PolicyNetwork, ValueNetwork

        run_id = st.selectbox("Select Run", [r['run_id'] for r in reinforce_runs])
        run_meta = next(r for r in reinforce_runs if r['run_id'] == run_id)
        hp = run_meta.get('hyperparameters', {})

        hidden_dim = int(hp.get('hidden_dim', 64))
        checkpoint = torch.load(
            os.path.join(RUNS_DIR, f"{run_id}.pt"),
            map_location='cpu',
            weights_only=True,
        )

        policy_net = PolicyNetwork(obs_dim=10, action_dim=5, hidden_dim=hidden_dim)
        value_net = ValueNetwork(obs_dim=10, hidden_dim=hidden_dim)
        policy_net.load_state_dict(checkpoint['policy_state_dict'])
        value_net.load_state_dict(checkpoint['value_state_dict'])
        policy_net.eval()
        value_net.eval()

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("State Input")
            ret_1h = st.slider("1h Log Return", -0.10, 0.10, 0.0, 0.001, format="%.3f", key="reinforce_ret_1h")
            ret_4h = st.slider("4h Log Return", -0.20, 0.20, 0.0, 0.005, format="%.3f", key="reinforce_ret_4h")
            ret_24h = st.slider("24h Log Return", -0.50, 0.50, 0.0, 0.01, format="%.2f", key="reinforce_ret_24h")
            vol_24h = st.slider("Volatility (24h σ)", 0.0, 0.10, 0.02, 0.005, format="%.3f", key="reinforce_vol_24h")
            log_vol = st.slider("Log Volume", 10.0, 30.0, 20.0, 0.5, key="reinforce_log_vol")
            hour_sin = st.slider("Hour Sin", -1.0, 1.0, 0.0, 0.1, key="reinforce_hour_sin")
            hour_cos = st.slider("Hour Cos", -1.0, 1.0, 1.0, 0.1, key="reinforce_hour_cos")
            day_sin = st.slider("Day Sin", -1.0, 1.0, 0.0, 0.1, key="reinforce_day_sin")
            day_cos = st.slider("Day Cos", -1.0, 1.0, 1.0, 0.1, key="reinforce_day_cos")
            pos = st.slider("Portfolio Position", 0.0, 1.0, 0.0, 0.05, key="reinforce_pos")

        state = torch.FloatTensor([
            ret_1h, ret_4h, ret_24h, vol_24h, log_vol,
            hour_sin, hour_cos, day_sin, day_cos, pos
        ]).unsqueeze(0)

        with torch.no_grad():
            logits = policy_net(state)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            value_estimate = float(value_net(state).item())

        greedy = int(np.argmax(probs))

        with col2:
            st.metric("State Value Estimate", f"{value_estimate:.4f}")
            st.subheader("Policy Probabilities")
            colors = ['#ff6b6b' if i != greedy else '#51cf66' for i in range(5)]
            fig = go.Figure(go.Bar(
                x=ACTION_NAMES, y=probs,
                marker_color=colors,
                text=[f"{v:.3f}" for v in probs],
                textposition='outside',
            ))
            fig.update_layout(
                height=350, margin=dict(t=10, b=30),
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                annotations=[dict(
                    x=ACTION_NAMES[greedy], y=probs[greedy],
                    text="★ Greedy", showarrow=True, arrowhead=2,
                    yshift=25, font=dict(size=14, color='#51cf66')
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Policy Surface (Position × 1h Return)")
            pos_range = np.linspace(0, 1, 21)
            ret_range = np.linspace(-0.05, 0.05, 21)
            grid = np.zeros((len(ret_range), len(pos_range)), dtype=int)

            for i, r_val in enumerate(ret_range):
                for j, p_val in enumerate(pos_range):
                    s = torch.FloatTensor([
                        r_val, ret_4h, ret_24h, vol_24h, log_vol,
                        hour_sin, hour_cos, day_sin, day_cos, p_val
                    ]).unsqueeze(0)
                    with torch.no_grad():
                        grid[i, j] = policy_net(s).argmax().item()

            fig_h = go.Figure(go.Heatmap(
                z=grid,
                x=[f"{p:.2f}" for p in pos_range],
                y=[f"{r:.3f}" for r in ret_range],
                colorscale=[
                    [0.0, '#ff6b6b'], [0.25, '#ffa06b'],
                    [0.5, '#ffe66b'], [0.75, '#6bcfff'], [1.0, '#51cf66']
                ],
                zmin=0, zmax=4,
                colorbar=dict(
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=ACTION_NAMES,
                ),
                hovertemplate="Position: %{x}<br>Return 1h: %{y}<br>"
                              "Action: %{customdata}<extra></extra>",
                customdata=[[ACTION_NAMES[grid[i, j]] for j in range(len(pos_range))]
                            for i in range(len(ret_range))],
            ))
            fig_h.update_layout(
                xaxis_title="Position",
                yaxis_title="1h Log Return",
                height=400, margin=dict(t=10, b=30),
            )
            st.plotly_chart(fig_h, use_container_width=True)
