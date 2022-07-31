from environment import SingleEnvironment
from bandit_exp.agents.reinforcement_learners import QSoftmax, EpsilonGreedy
from bandit_tasks import BernoulliMultiArmedBandit

import streamlit as st
import numpy as np
import pandas as pd


st.header("Bandit Setting")

time_horizon = st.number_input("Time Horizon", 1, 100000)
means = st.text_input("Means for Each Arms")
means = [float(mean) for mean in means.split()]
n_arms = len(means)

# sds = st.text_input("Standard Deviation for Each Arm")
# sds = [float(sds) for sds in sds.split()]
initial_values = np.tile(0, n_arms)

bandit_task_option = st.selectbox("Bandit Task", ["Bernouli"])

st.header("Agent Setting")
agent_option = st.selectbox("Agent", ["Q learner", "epsilon greedy"])

if agent_option == "Q learner":
    learning_rate = st.slider("learning rate", 0.001, 1.0)
    inverse_temperature = st.slider("inverse temperature", -10.0, 10.0)
    agent = QSoftmax(learning_rate, inverse_temperature, initial_values)
elif agent_option == "epsilon greedy":
    epsilon = st.slider("Epsilon", 0.0, 0.99)
    agent = EpsilonGreedy(epsilon, initial_values)

if bandit_task_option == "Bernouli":
    bandit_task = BernoulliMultiArmedBandit(means)

if st.button("Start simulating"):
    latest_iteration = st.empty()
    bar = st.progress(0)
    env = SingleEnvironment(
        time_horizon, agent, bandit_task, "./treamlist_history.json"
    )
    for i in range(time_horizon):
        env.run_simulation()
        latest_iteration.text(f"Iteration {i+1}")
        bar.progress(min(i / time_horizon, 1.0))

    st.header("Result")
    hist = env.history["estimated_reward"]
    df = pd.DataFrame(hist, columns=means)
    st.line_chart(df)
