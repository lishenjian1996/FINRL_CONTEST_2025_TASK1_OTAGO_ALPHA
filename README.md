# FINRL_CONTEST_2025_TASK1_OTAGO_ALPHA

For back testing:

Because we modified input indicators for improve it's performance.

We have insert these codes for runing backtest successfully.

####################
# In order to be replicate our result, we need to fix seeds
import os
import random
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#####################

#####################
# === Create new dataset specifically for PPO model with 1093-dimensional state ===
trade_putcall = trade.copy()  # Create a copy of the original DataFrame

# Ensure there are no missing values in the columns of interest
for col in ['vix', 'turbulence', 'PutCallRatio']:
    if col not in trade_putcall.columns:
        trade_putcall[col] = 0
    trade_putcall[col] = pd.to_numeric(trade_putcall[col], errors='coerce').fillna(0)
######################


######################
# Load the trade data from 2_2_trade_data_with_putcall_2019_2023.csv, with added indicator for trade
trade = pd.read_csv('2_2_trade_data_with_putcall_2019_2023.csv')

# Convert to pandas DataFrame
trade = pd.DataFrame(dataset['train'])

trade = trade.drop('Unnamed: 0',axis=1)

# Create a new index based on unique dates
unique_dates = trade['date'].unique()
date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}

# Create new index based on the date mapping
trade['new_idx'] = trade['date'].map(date_to_idx)

# Set this as the index
trade = trade.set_index('new_idx')
#######################

#######################
# === For PPO model with 1093-dimensional state ===
stock_dimension_putcall = len(trade.tic.unique())
state_space_putcall = 1 + 2 * stock_dimension_putcall + (len(INDICATORS) + 3) * stock_dimension_putcall  # 

print(f"Stock Dimension: {stock_dimension_putcall}, State Space: {state_space_putcall}")
#######################

#######################
# Define trading environment parameters for putcall-based PPO model
buy_cost_list_putcall = sell_cost_list_putcall = [0.001] * stock_dimension_putcall
num_stock_shares_putcall = [0] * stock_dimension_putcall
INDICATORS_putcall = INDICATORS + ['vix', 'turbulence','PutCallRatio']
INDICATORS_putcall = list(dict.fromkeys(INDICATORS_putcall))

env_kwargs_putcall = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares_putcall,
    "buy_cost_pct": buy_cost_list_putcall,
    "sell_cost_pct": sell_cost_list_putcall,
    "state_space": state_space_putcall,
    "stock_dim": stock_dimension_putcall,
    "tech_indicator_list": INDICATORS_putcall,
    "action_space": stock_dimension_putcall,
    "reward_scaling": 1e-4
}
########################

########################
env = StockTradingEnv(df=trade_putcall, **env_kwargs_putcall)
state, _ = env.reset()
print(len(state))   # Ensure the state space is 1093
########################


########################
# Create environment for putcall-based PPO model
e_trade_gym_putcall = StockTradingEnv(df=trade_putcall, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs_putcall)
state, _ = e_trade_gym_putcall.reset()
print("Test state dimension:", len(state))  # Ensure the state space is 1093
print("env_kwargs_putcall state space:", env_kwargs_putcall["state_space"]) # state space should be 1093
########################

########################
# === For putcall PPO model ===
observation_space_putcall = e_trade_gym_putcall.observation_space
action_space_putcall = e_trade_gym_putcall.action_space

print("State shape:", observation_space_putcall.shape) # state space should be 1093
########################

########################
# Load the PPO putcall model
loaded_ppo = MLPActorCritic(observation_space=observation_space_putcall, action_space=action_space_putcall, hidden_sizes=(512, 512))
loaded_ppo.load_state_dict(torch.load('./trained_models/agent_ppo_100_epochs_20k_steps.pth'))
loaded_ppo.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
loaded_ppo.eval() # Set the model to evaluation mode
########################

########################
# Test Model by using modified environment
df_assets_ppo, df_account_value_ppo, df_actions_ppo, df_portfolio_distribution_ppo = DRL_prediction(act=loaded_ppo, environment=e_trade_gym_putcall)

print("Test env state length:", len(e_trade_gym.state)) # should be 841
print("Expected input dim for model:", loaded_ppo.pi.mu_net[0].in_features) # should be 1093
########################

Rest of code keeps the same with default code as the df_assets_ppo, df_account_value_ppo, df_actions_ppo, df_portfolio_distribution_ppo has been tested in the modified environment: e_trade_gym_putcall



