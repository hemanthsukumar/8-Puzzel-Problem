import sys
import itertools
import argparse
import time
import heapq

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('start', type=str, help='the start state file name')
parser.add_argument('goal', type=str, help='the goal state file name')
parser.add_argument('method', type=str, help='the search method to use')
parser.add_argument('flag', type=str, default = False, help='the flag to use for search method')
args = parser.parse_args()

start_file = args.start
goal_file = args.goal
method = args.method
flag = args.flag

class Node:
    def __init__(self, state, parent=None, cost=0, depth=0, g_score=0, h_score=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.depth = depth
        self.g_score = g_score
        self.h_score = h_score
        self.f_score = g_score + h_score
        self.move_cost = 0
    def __lt__(self, other):
        return self.cost + self.h_score < other.cost + other.h_score
    def __eq__(self, other):
        return self.state == other.state
    def __hash__(self):
        return hash(tuple(map(tuple, self.state)))

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        state = []
        for line in f:
            if line.strip() == 'END OF FILE':
                break
            state.append([int(x) for x in line.split()])
    return state

def write_file(file_name, frontier, explored, generated, path_to_goal, moves, cost_of_path, time_taken, popped,flag, b):
    
    print(f"Nodes Popped: {popped}\n")
    print(f"Nodes Expanded: {len(explored)}\n")
    print(f"Nodes Generated: {generated}\n")
    print(f"Max Fringe Size: {len(frontier)}\n")
    if b == 0:
        print(f"Solution found at depth {moves} with cost of {cost_of_path}.\n")
    else:
        print(f"Solution found at depth {moves}.\n")
        
    if moves != -1:
        curr_state = path_to_goal[0]
        zero_row, zero_col = find_zero(curr_state)
        print("Steps:\n")
        for i in range(1, len(path_to_goal)):
            next_state = path_to_goal[i]
            next_zero_row, next_zero_col = find_zero(next_state)
            curr_state = path_to_goal[i]
            move_num = i
            move_dir = ''
            if next_zero_row > zero_row:
                move_dir = 'Down'
            elif next_zero_row < zero_row:
                move_dir = 'Up'
            elif next_zero_col > zero_col:
                move_dir = 'Right'
            elif next_zero_col < zero_col:
                move_dir = 'Left'
            print(f"\tMove {next_state[zero_row][zero_col]} {move_dir}\n")
            zero_row, zero_col = next_zero_row, next_zero_col

    if flag == "True" :
        with open(file_name, 'w') as f:
            f.write(f"Command-Line Arguments : ['start.txt', 'goal.txt', '{method}', '{flag}']\n")
            f.write(f"Method selected: {method}\n")
            f.write(f"Running {method}\n")
            f.write(f"path_to_goal: {path_to_goal}\n")
            f.write(f"cost_of_path: {cost_of_path}\n")
            f.write(f"nodes_expanded: {explored}\n")
            f.write(f"nodes_generated: {generated}\n")
            f.write(f"search_depth: {moves}\n")
            f.write(f"max_search_depth: {max([node.depth for node in frontier])}\n")
            f.write(f"running_time: {round(time_taken, 8)}\n")
            # Write moves to file
            if moves != -1:
                curr_state = path_to_goal[0]
                zero_row, zero_col = find_zero(curr_state)
                f.write(f'Starting state:\n{curr_state[0]}\n{curr_state[1]}\n{curr_state[2]}\n')
                for i in range(1, len(path_to_goal)):
                    next_state = path_to_goal[i]
                    next_zero_row, next_zero_col = find_zero(next_state)
                    for row in range(len(curr_state)):
                        for col in range(len(curr_state[row])):
                            if curr_state[row][col] == 0:
                                f.write(f'\t{next_state[zero_row][zero_col]}')
                            elif col > 0 and curr_state[row][col] == next_state[row][col-1]:
                                f.write(f'\t{next_state[row][col]}')
                            elif col < 2 and curr_state[row][col] == next_state[row][col+1]:
                                f.write(f'\t{next_state[row][col]}')
                            elif row > 0 and curr_state[row][col] == next_state[row-1][col]:
                                f.write(f'\t{next_state[row][col]}')
                            elif row < 2 and curr_state[row][col] == next_state[row+1][col]:
                                f.write(f'\t{next_state[row][col]}')
                            else:
                                f.write(f'\tX')
                        f.write('\n')
                    f.write('\n')
                    curr_state = path_to_goal[i]
                    move_num = i
                    move_dir = ''
                    if next_zero_row > zero_row:
                        move_dir = 'Down'
                    elif next_zero_row < zero_row:
                        move_dir = 'Up'
                    elif next_zero_col > zero_col:
                        move_dir = 'Right'
                    elif next_zero_col < zero_col:
                        move_dir = 'Left'
                    f.write(f"Move {move_num}: {next_state[zero_row][zero_col]} {move_dir}\n")
                    zero_row, zero_col = next_zero_row, next_zero_col
                f.write(f"Nodes Popped: {popped}\n")
                f.write(f"Nodes Expanded: {len(explored)}\n")
                f.write(f"Nodes Generated: {generated}\n")
                f.write(f"Max Fringe Size: {len(frontier)}\n")    

def find_zero(state):
    for row in range(len(state)):
        for col in range(len(state[row])):
            if state[row][col] == 0:
                return row, col

def heuristic_misplaced_tiles(state, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                flat_goal_state = list(itertools.chain.from_iterable(goal))
                row, col = divmod(flat_goal_state.index(state[i][j]), 3)
                distance += abs(i - row) + abs(j - col)
    return distance

def get_successors(node, goal):
    successors = []
    zero_row, zero_col = find_zero(node.state)

    if zero_row > 0:
        up_state = [x[:] for x in node.state]
        up_state[zero_row][zero_col], up_state[zero_row-1][zero_col] = up_state[zero_row-1][zero_col], up_state[zero_row][zero_col]
        up_cost = up_state[zero_row-1][zero_col]
        up_g_score = node.g_score + up_cost
        up_h_score = heuristic_misplaced_tiles(up_state, goal)
        successors.append(Node(up_state, node, up_g_score, node.depth + 1, up_h_score))
        successors[-1].move_cost = up_cost

    if zero_row < 2:
        down_state = [x[:] for x in node.state]
        down_state[zero_row][zero_col], down_state[zero_row+1][zero_col] = down_state[zero_row+1][zero_col], down_state[zero_row][zero_col]
        down_cost = down_state[zero_row+1][zero_col]
        down_g_score = node.g_score + down_cost
        down_h_score = heuristic_misplaced_tiles(down_state, goal)
        successors.append(Node(down_state, node, down_g_score, node.depth + 1, down_h_score))
        successors[-1].move_cost = down_cost

    if zero_col > 0:
        left_state = [x[:] for x in node.state]
        left_state[zero_row][zero_col], left_state[zero_row][zero_col-1] = left_state[zero_row][zero_col-1], left_state[zero_row][zero_col]
        left_cost = left_state[zero_row][zero_col-1]
        left_g_score = node.g_score + left_cost
        left_h_score = heuristic_misplaced_tiles(left_state, goal)
        successors.append(Node(left_state, node, left_g_score, node.depth + 1, left_h_score))
        successors[-1].move_cost = left_cost

    if zero_col < 2:
        right_state = [x[:] for x in node.state]
        right_state[zero_row][zero_col], right_state[zero_row][zero_col+1] = right_state[zero_row][zero_col+1], right_state[zero_row][zero_col]
        right_cost = right_state[zero_row][zero_col+1]
        right_g_score = node.g_score + right_cost
        right_h_score = heuristic_misplaced_tiles(right_state, goal)
        successors.append(Node(right_state, node, right_g_score, node.depth + 1, right_h_score))
        successors[-1].move_cost = right_cost

    return successors



def astar(initial_state, goal_state, flag):
    b = 0
    start_time = time.time()
    initial_node = Node(initial_state, None, 0, 0, heuristic_misplaced_tiles(initial_state, goal_state))
    explored = set()
    generated = 0
    frontier = [initial_node]
    heapq.heapify(frontier)
    popped = 0
    while frontier:
        node = heapq.heappop(frontier)
        popped += 1
        explored.add(tuple(tuple(row) for row in node.state))
        if node.state == goal_state:
            path_to_goal = []
            curr_node = node
            while curr_node is not None:
                path_to_goal.append(curr_node.state)
                curr_node = curr_node.parent
            path_to_goal.reverse()
            moves = len(path_to_goal) - 1
            cost_of_path = node.cost
            time_taken = time.time() - start_time
            write_file("output.txt", frontier, explored, generated, path_to_goal, moves, cost_of_path, time_taken, popped, flag, b)
            return
        successors = get_successors(node, goal_state)
        generated += len(successors)
        for child_node in successors:
            child_g_score = node.g_score + child_node.g_score
            child_h_score = heuristic_misplaced_tiles(child_node.state, goal_state)
            child_f_score = child_g_score + child_h_score
            child_node = Node(child_node.state, node, child_g_score, child_h_score, child_f_score)
            if tuple(tuple(row) for row in child_node.state) not in explored:
                heapq.heappush(frontier, child_node)
    write_file("output.txt", frontier, explored, generated, [], -1, -1, time.time() - start_time, popped, flag, b)

def bfs(initial_state, goal_state, flag):
    b=1
    start_time = time.time()
    initial_node = Node(initial_state)
    frontier = [initial_node]
    explored = set()
    generated = 0
    popped=0
    while len(frontier) != 0:
        node = frontier.pop(0)
        explored.add(tuple(map(tuple, node.state)))
        popped+=1
        if node.state == goal_state:
            path_to_goal = []
            curr_node = node
            while curr_node is not None:
                path_to_goal.append(curr_node.state)
                curr_node = curr_node.parent
            path_to_goal.reverse()
            moves = len(path_to_goal) - 1
            cost_of_path = node.cost
            time_taken = time.time() - start_time
            write_file("output.txt", frontier, explored, generated, path_to_goal, moves, cost_of_path, time_taken, popped, flag, b)
            return
        successors = get_successors(node, goal_state)
        generated += len(successors)
        for succ in successors:
            if tuple(tuple(row) for row in succ.state) not in explored:
                frontier.append(succ)
                explored.add(tuple(tuple(row) for row in succ.state))
    write_file("output.txt", frontier, explored, generated, None, None, None, None, popped, flag, b)


def dfs(initial_state, goal_state, flag):
    b = 1
    start_time = time.time()
    initial_node = Node(initial_state)
    frontier = [initial_node]
    explored = set()
    generated = 0
    popped = 0
    while len(frontier) != 0:
        node = frontier.pop()
        explored.add(tuple(tuple(row) for row in node.state))
        popped+=1
        if node.state == goal_state:
            path_to_goal = []
            curr_node = node
            while curr_node is not None:
                path_to_goal.append(curr_node.state)
                curr_node = curr_node.parent
            path_to_goal.reverse()
            moves = len(path_to_goal) - 1
            cost_of_path = node.cost
            print("Cost of path:", cost_of_path)
            print("H value of goal state:", heuristic_misplaced_tiles(node.state, goal_state))
            time_taken = time.time() - start_time
            write_file("output.txt", frontier, explored, generated, path_to_goal, moves, cost_of_path, time_taken, popped, flag, b)
            return
        successors = get_successors(node, goal_state)
        generated += len(successors)
        for succ in successors[::-1]:
            if tuple(tuple(row) for row in succ.state) not in explored:
                frontier.append(succ)
                explored.add(tuple(tuple(row) for row in node.state))
    write_file("output.txt", frontier, explored, generated, None, None, None, None, popped, flag, b)

def ids(initial_state, goal_state, flag):
    b = 1
    start_time = time.time()
    initial_node = Node(initial_state)
    explored = set()
    generated = 0
    popped = 0
    depth_limit = 0
    while True:
        frontier = [initial_node]
        curr_depth = 0
        while len(frontier) != 0:
            node = frontier.pop()
            explored.add(tuple(tuple(row) for row in node.state))
            popped += 1
            if node.state == goal_state:
                path_to_goal = []
                curr_node = node
                while curr_node is not None:
                    path_to_goal.append(curr_node.state)
                    curr_node = curr_node.parent
                path_to_goal.reverse()
                moves = len(path_to_goal) - 1
                cost_of_path = node.cost
                time_taken = time.time() - start_time
                write_file("output.txt", frontier, explored, generated, path_to_goal, moves, cost_of_path, time_taken, popped, flag, b)
                return
            if curr_depth < depth_limit and node.depth < depth_limit:
                successors = get_successors(node, goal_state)
                generated += len(successors)
                for succ in successors:
                    if tuple(tuple(row) for row in succ.state) not in explored and succ.depth <= depth_limit:
                        frontier.append(succ)
        depth_limit += 1
        curr_depth += 1
        explored.clear()


def greedy(initial_state, goal_state, flag):
    b = 0
    start_time = time.time()
    initial_node = Node(initial_state, None, 0, 0, heuristic_misplaced_tiles(initial_state, goal_state))
    explored = set()
    generated = 0
    frontier = [initial_node]
    heapq.heapify(frontier)
    popped = 0
    while len(frontier) != 0:
        node = heapq.heappop(frontier)
        explored.add(tuple(tuple(row) for row in node.state))
        popped+=1
        if node.state == goal_state:
            path_to_goal = []
            curr_node = node
            while curr_node is not None:
                path_to_goal.append(curr_node.state)
                curr_node = curr_node.parent
            path_to_goal.reverse()
            moves = len(path_to_goal) - 1
            cost_of_path = heuristic_misplaced_tiles(initial_state, goal_state)
            time_taken = time.time() - start_time
            write_file("output.txt", frontier, explored, generated, path_to_goal, moves, cost_of_path, time_taken, popped, flag, b)
            return
        for successor in get_successors(node, goal_state):
            generated += 1
            if tuple(tuple(row) for row in successor.state) not in explored:
                successor.heuristic = heuristic_misplaced_tiles(successor.state, goal_state)
                heapq.heappush(frontier, successor)
    write_file("output.txt", frontier, explored, generated, [], -1, -1, time.time() - start_time, popped, flag, b)


def ucs(initial_state, goal_state, flag):
    b = 0
    start_time = time.time()
    initial_node = Node(initial_state, None, 0, 0, 0)
    explored = set()
    generated = 0
    frontier = [(0, initial_node)]
    heapq.heapify(frontier)
    frontier_nodes = [initial_node] 
    popped = 0
    cost_of_path = 0
    while frontier:
        node = heapq.heappop(frontier)[1]
        explored.add(tuple(tuple(row) for row in node.state))
        frontier_nodes.remove(node)  
        popped += 1
        
        if node.state == goal_state:
            path_to_goal = []
            curr_node = node
            while curr_node is not None:
                path_to_goal.append(curr_node.state)
                curr_node = curr_node.parent
            path_to_goal.reverse()
            moves = len(path_to_goal) - 1
            cost_of_path = 0
            curr_node = node  
            while curr_node is not None:
                if curr_node.move_cost is not None:  
                    cost_of_path += curr_node.move_cost
                curr_node = curr_node.parent
            time_taken = time.time() - start_time
            write_file("output.txt", frontier_nodes, explored, generated, path_to_goal, moves, cost_of_path, time_taken, popped, flag, b)
            return
        successors = get_successors(node, goal_state) 
        generated += len(successors)
        for a in successors:
            child_g_score = node.g_score + a.move_cost  
            child_node = Node(a.state, node, a.move_cost, node.depth + 1, child_g_score)
            child_tuple = (child_g_score, child_node)
            if tuple(tuple(row) for row in child_node.state) not in explored:
                heapq.heappush(frontier, child_tuple)
                frontier_nodes.append(child_node)  
            elif tuple(tuple(row) for row in child_node.state) in explored:
                for index, (priority, old_node) in enumerate(frontier):
                    if tuple(tuple(row) for row in old_node.state) == tuple(tuple(row) for row in child_node.state):
                        if old_node.g_score > child_node.g_score:
                            frontier[index] = (child_node.g_score, child_node)
                            heapq.heapify(frontier)
                            frontier_nodes[index] = child_node  
                            break
    write_file("output.txt", frontier_nodes, explored, generated, [], -1, -1, time.time() - start_time, popped, flag, b)


def dls(initial_state, goal_state, flag):
    b=1
    depth_limit = int(input("Enter depth limit: "))
    start_time = time.time()
    initial_node = Node(initial_state, None, 0, 0, heuristic_misplaced_tiles(initial_state, goal_state))
    explored = set()
    generated = 0
    frontier = [initial_node]
    heapq.heapify(frontier)
    popped = 0
    while frontier:
        if not frontier:
            break
        node = heapq.heappop(frontier)
        explored.add(tuple(tuple(row) for row in node.state))
        popped+=1
        if node.state == goal_state:
            path_to_goal = []
            curr_node = node
            while curr_node is not None:
                path_to_goal.append(curr_node.state)
                curr_node = curr_node.parent
            path_to_goal.reverse()
            moves = len(path_to_goal) - 1
            cost_of_path = node.cost
            time_taken = time.time() - start_time
            write_file("output.txt", frontier, explored, generated, path_to_goal, moves, cost_of_path, time_taken, popped, flag, b=1)
            return
        if node.depth > depth_limit:
            break
        if node.depth < depth_limit:
            successors = get_successors(node, goal_state)
            generated += len(successors)
            
            for succ in successors:
                child_g_score = node.g_score + succ.move_cost
                child_h_score = heuristic_misplaced_tiles(succ.state, goal_state)
                child_f_score = child_g_score + child_h_score
                child_node = Node(succ.state, node, child_g_score, child_h_score, child_f_score)
                if tuple(tuple(row) for row in child_node.state) not in explored:
                    heapq.heappush(frontier, child_node)
    write_file("output.txt", frontier, explored, generated, [], -1, -1, time.time() - start_time, popped, flag, b=1)




if __name__ == "__main__":
    start_state = read_file(start_file)
    goal_state = read_file(goal_file)
    if method == "astar":
        astar(start_state, goal_state, flag)
    elif method == "bfs":
        bfs(start_state, goal_state, flag)
    elif method == "dfs":
        dfs(start_state, goal_state, flag)
    elif method == "ids":
        ids(start_state, goal_state, flag)
    elif method == "greedy":
        greedy(start_state, goal_state, flag)
    elif method == "ucs":
        ucs(start_state, goal_state, flag)
    elif method == "dls":
        dls(start_state, goal_state, flag)
    else:
        print("Enter method name from the following: astar\tbfs\tdfs\tids\tgreedy\tucs\tdls")
        
    
    

    
    
