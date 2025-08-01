import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Helper functions for Blackjack
def draw_card():
    return random.choice([1,2,3,4,5,6,7,8,9,10,10,10,10])

def draw_hand():
    return [draw_card(), draw_card()]

def usable_ace(hand):
    # Usable ace: counts as 11, but avoids bust
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    # Return current hand total, counting ace as 11 if it doesn't bust
    s = sum(hand)
    if 1 in hand and s + 10 <= 21:
        return s + 10
    return s

def is_bust(hand):
    return sum_hand(hand) > 21

def score(hand):
    s = sum_hand(hand)
    return 0 if is_bust(hand) else s

def dealer_policy(hand):
    # Dealer hits until sum >= 17
    while sum_hand(hand) < 17:
        hand.append(draw_card())
    return hand

def get_state(hand, dealer_card):
    return (sum_hand(hand), dealer_card, usable_ace(hand))

# Policy: stick on 20, 21; hit otherwise (as per Example 5.1)
def player_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 'stick'
    else:
        return 'hit'

# Episode simulation: returns (states, reward)
def generate_episode(policy):
    # Initialize hands
    player = draw_hand()
    dealer = draw_hand()
    dealer_card = dealer[0]

    states, actions = [], []

    while True:
        player_sum = sum_hand(player)
        if player_sum < 12:
            # Always hit on less than 12
            player.append(draw_card())
            continue

        state = (player_sum, dealer_card, usable_ace(player))
        action = policy(state)
        states.append(state)
        actions.append(action)

        if action == 'stick':
            break
        # player's action: hit
        player.append(draw_card())
        if is_bust(player):
            return states, -1  # Player busts immediately

    # Dealer's turn
    dealer = dealer_policy(dealer)
    player_score = score(player)
    dealer_score = score(dealer)
    if is_bust(dealer):
        return states, 1
    elif dealer_score > player_score:
        return states, -1
    elif dealer_score < player_score:
        return states, 1
    else:
        return states, 0

# Monte Carlo policy evaluation (first visit)
def mc_prediction(policy, num_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)

    for i in range(num_episodes):
        print(f'Episode: {i}')
        states, reward = generate_episode(policy)
        visited = set()
        for idx, state in enumerate(states):
            # Only first visit
            if state not in visited:
                visited.add(state)
                # All rewards are at end, so G = reward; gamma=1.0
                returns_sum[state] += reward
                returns_count[state] += 1

    for state in returns_sum:
        V[state] = returns_sum[state] / returns_count[state]
    return V

# Prepare value function for plotting
def plot_value_function(V, title="Value Function"):
    # Only show states player_sum 12-21, dealer showing 1-10
    player_range = range(12, 22)
    dealer_range = range(1, 11)
    X, Y = np.meshgrid(dealer_range, player_range)

    def compute_Z(usable_ace):
        Z = np.apply_along_axis(
            lambda x: V.get((int(x[1]), int(x[0]), usable_ace), 0),
            2, np.dstack([X, Y])
        )
        return Z
    # Plot for usable ace = True
    fig = plt.figure(figsize=(12, 5))
    for i, usable in enumerate([True, False]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        Z = compute_Z(usable)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.viridis, edgecolor='none')
        ax.set_title(f'{title}\nUsable Ace: {usable}')
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_zlabel('Value')
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(12, 22))
        ax.set_zlim(-1, 1)
        fig.colorbar(surf, shrink=0.5, aspect=5, ax=ax)
    plt.tight_layout()
    plt.show()

def print_policy(policy_fn, title="Policy"):
    player_range = range(12, 22)
    dealer_range = range(1, 11)

    def policy_symbol(state):
        action = policy_fn(state)
        return 'S' if action == 'stick' else 'H'

    print(f"\n{title} (Usable Ace)")
    print("    " + " ".join([f"{d:2d}" for d in dealer_range]))
    for player_sum in reversed(player_range):
        row = [policy_symbol((player_sum, dealer, True)) for dealer in dealer_range]
        print(f"{player_sum:2d}: " + " ".join(row))

    print(f"\n{title} (No Usable Ace)")
    print("    " + " ".join([f"{d:2d}" for d in dealer_range]))
    for player_sum in reversed(player_range):
        row = [policy_symbol((player_sum, dealer, False)) for dealer in dealer_range]
        print(f"{player_sum:2d}: " + " ".join(row))

if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    num_episodes = 500000
    print("Running Monte Carlo first-visit policy evaluation...")
    V = mc_prediction(player_policy, num_episodes)
    plot_value_function(V, "State-Value Function under Policy (stick>=20)")
    print_policy(player_policy, "Player Policy (S=stick, H=hit)")

# if __name__ == '__main__':
#     np.random.seed(42)
#     random.seed(42)
#     num_episodes = 500000
#     print("Running Monte Carlo first-visit policy evaluation...")
#     V = mc_prediction(player_policy, num_episodes)
#     plot_value_function(V, "State-Value Function under Policy (stick>=20)")
