import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# ------------ Blackjack environment ------------
def draw_card():
    return random.choice([1,2,3,4,5,6,7,8,9,10,10,10,10])

def draw_hand():
    return [draw_card(), draw_card()]

def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    s = sum(hand)
    return s + 10 if usable_ace(hand) else s

def is_bust(hand):
    return sum_hand(hand) > 21

def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

def dealer_policy(hand):
    while sum_hand(hand) < 17:
        hand.append(draw_card())
    return hand

def get_state(hand, dealer_card):
    return (sum_hand(hand), dealer_card, usable_ace(hand))

ACTIONS = ['stick', 'hit']

# ------------ MCES for Blackjack ------------

def get_possible_start_states():
    # player's sum: 12-21, dealer's card: 1-10, usable ace: T/F
    states = []
    for psum in range(12,22):
        for dealer in range(1,11):
            for ace in [True, False]:
                states.append((psum, dealer, ace))
    return states

def starting_hand_from_state(player_sum, usable_ace):
    """
    Find a possible 2-card hand for the given player_sum and usable_ace status.
    """
    hands = []
    for card1 in range(1, 11):
        for card2 in range(1, 11):
            if card1 == 1 or card2 == 1:
                s = card1 + card2 + 10 if card1 + card2 + 10 <= 21 else card1 + card2
                ace_flag = (1 in (card1, card2)) and (s + 10 <= 21)
            else:
                s = card1 + card2
                ace_flag = False
            if s == player_sum and ace_flag == usable_ace and (card1, card2) != (1,1):
                hands.append([card1, card2])
    # if multiple hands possible, choose one
    return random.choice(hands) if hands else None

def generate_episode_ES(policy, Q, epsilon=0.1):
    """
    Generate an episode using exploring starts: 
    Random valid initial state and action.
    """
    # random start state and action
    player_sum = random.randint(12,21)
    dealer_card = random.randint(1,10)
    usable = bool(random.getrandbits(1))
    player_hand = starting_hand_from_state(player_sum, usable)
    # If somehow impossible, fall back to a default random hand
    if player_hand is None: player_hand = draw_hand()
    dealer_hand = [dealer_card, draw_card()]
    state = (sum_hand(player_hand), dealer_card, usable_ace(player_hand))

    # randomly choose initial action
    action = random.choice(ACTIONS)
    episode = []
    episode.append((state, ACTIONS.index(action), 0))  # (state, action, reward)

    # player's turn
    while True:
        if action == 'stick':
            break
        # hit
        player_hand.append(draw_card())
        if is_bust(player_hand):
            state_ = (sum_hand(player_hand), dealer_card, usable_ace(player_hand))
            episode.append((state_, None, -1))
            return episode
        state_ = (sum_hand(player_hand), dealer_card, usable_ace(player_hand))
        # choose action according to current policy, epsilon-soft
        probs = np.ones(len(ACTIONS)) * epsilon / len(ACTIONS)
        best_a = np.argmax([Q.get((state_, a), 0) for a in range(2)])
        probs[best_a] += 1 - epsilon
        action = np.random.choice(ACTIONS, p=probs)
        episode.append((state_, ACTIONS.index(action), 0))
        state = state_
    
    # dealer's turn
    dealer_hand = dealer_policy(dealer_hand)
    player_score = score(player_hand)
    dealer_score = score(dealer_hand)
    if is_bust(dealer_hand):
        reward = 1
    elif dealer_score > player_score:
        reward = -1
    elif dealer_score < player_score:
        reward = 1
    else:
        reward = 0
    # Set reward for last step in episode
    episode[-1] = (episode[-1][0], episode[-1][1], reward)
    return episode

def MCES_Blackjack(num_episodes=500000, epsilon=0.1):
    Q = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    policy = {}
    for i in range(num_episodes):
        print(f'Episode: {i}')
        episode = generate_episode_ES(policy, Q, epsilon)
        # find first occurrence of each (s,a)
        sa_in_episode = set()
        for t, (state, action, reward) in enumerate(episode):
            if action is None: continue  # bust terminal step
            if (state, action) not in sa_in_episode:
                sa_in_episode.add((state, action))
                # G is the reward from that time onwards (here, only final reward at end)
                G = episode[-1][2]
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                Q[(state, action)] = returns_sum[(state, action)] / returns_count[(state, action)]

        # policy improvement (after episode)
        for state, _, _ in episode:
            q_stick = Q.get((state, 0), 0)
            q_hit = Q.get((state, 1), 0)
            if q_stick > q_hit:
                policy[state] = 0  # stick
            else:
                policy[state] = 1  # hit
    return policy, Q

# ----- Visualization -----
def plot_policy_bad(policy, title="Policy"):
    dealer_range = range(1, 11)
    player_range = range(12, 22)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for idx, usable_ace in enumerate([True, False]):
        policy_grid = np.zeros((10, 10), dtype='<U1')
        for i, player in enumerate(player_range):
            for j, dealer in enumerate(dealer_range):
                state = (player, dealer, usable_ace)
                action = policy.get(state, 1)  # default to hit
                symbol = 'S' if action == 0 else 'H'
                policy_grid[9 - (player - 12), j] = symbol
        ax = axes[idx]
        im = ax.imshow(policy_grid, cmap=plt.cm.Pastel1, vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, policy_grid[i, j], ha="center", va="center", color="k", fontsize=16)
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels(dealer_range)
        ax.set_yticklabels(reversed(player_range))
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_title(f'Usable Ace: {usable_ace}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_policy(policy, title="Policy"):
    dealer_range = range(1, 11)
    player_range = range(12, 22)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for idx, usable_ace in enumerate([True, False]):
        policy_grid = np.zeros((10, 10), dtype=int)  # 0 for stick, 1 for hit
        text_grid = np.empty((10, 10), dtype='<U1')
        for i, player in enumerate(player_range):
            for j, dealer in enumerate(dealer_range):
                state = (player, dealer, usable_ace)
                action = policy.get(state, 1)  # default to hit
                symbol = 'S' if action == 0 else 'H'
                policy_grid[9 - (player - 12), j] = action
                text_grid[9 - (player - 12), j] = symbol
        ax = axes[idx]
        im = ax.imshow(policy_grid, cmap=plt.cm.Pastel1, vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, text_grid[i, j], ha="center", va="center", color="k", fontsize=16)
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels(dealer_range)
        ax.set_yticklabels(reversed(player_range))
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_title(f'Usable Ace: {usable_ace}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_value_function(Q, title="Optimal Value Function"):
    import matplotlib.pyplot as plt
    import numpy as np

    player_range = np.arange(12, 22)
    dealer_range = np.arange(1, 11)

    def get_V(usable_ace):
        V = np.zeros((len(player_range), len(dealer_range)))
        for i, player_sum in enumerate(player_range):
            for j, dealer_card in enumerate(dealer_range):
                state = (player_sum, dealer_card, usable_ace)
                # take max over actions for each state
                v = max(Q.get((state, 0), 0), Q.get((state, 1), 0))
                V[i, j] = v
        return V

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 5))
    for idx, usable_ace in enumerate([True, False]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        X, Y = np.meshgrid(dealer_range, player_range)
        Z = get_V(usable_ace)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.viridis, edgecolor='none')
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Value")
        ax.set_title(f"{title}\nUsable Ace: {usable_ace}")
        ax.view_init(30, -120)
        fig.colorbar(surf, shrink=0.5, aspect=5, ax=ax)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ----- Run everything -----
if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)

    print("Learning the optimal policy with MCES (on-policy control with exploring starts)...")
    policy, Q = MCES_Blackjack(num_episodes=1_000_000, epsilon=0.1)

    print("Plotting optimal policy (S = stick, H = hit)...")
    plot_policy(policy, title="Optimal Policy (Monte Carlo ES)")

    plot_value_function(Q)
