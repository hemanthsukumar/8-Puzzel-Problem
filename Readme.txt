README

Description:
This program is a command-line utility called "expense_8_puzzle" that solves the 8-puzzle problem using various search algorithms. The user provides a start state and a goal state for the 8-puzzle problem, along with the desired search method and a dump flag that indicates whether or not to output the search results to a file.

Usage:

Command line invocation format:
expense_8_puzzle.py <start-file> <goal-file> <method> <dump-flag>

Where:
* <start-file> is a text file containing the start state of the 8-puzzle problem.
* <goal-file> is a text file containing the goal state of the 8-puzzle problem.
* <method> is the search method to be used. Available options are:
   * bfs - Breadth-First Search
   * dfs - Depth-First Search
   * dls - Depth-Limited Search
   * ucs - Uniform-Cost Search
   * ids - Iterative Deepening Search
   * astar - A* Search
   * greedy - Greedy Best-First Search
* <dump-flag> is a flag indicating whether or not to output the search results to a file. Available options are:
   * True - Do not output results to a file.
   * False - Output results to a file.
	file is named as output.txt

Main Module:
expense_8_puzzle.py
start.txt
goal.txt
Readme.txt
After execution output.txt will be generated if flag is True 
Note: All files must be in same folder

Structure:
The code is contained in a single file called expense_8_puzzle.py. Here is a brief description of the main methods and classes:
* Node class: Represents a node in the search tree, with a state, a parent, a cost, and a depth.
* read_file function: Reads a text file and returns its contents as a list of integers.
* write_file function: Writes a string to a text file.
* find_zero function: Finds the position of the zero (empty tile) in a given state.
* heuristic_misplaced_tiles function: Computes the heuristic value of a state using the Misplaced Tiles heuristic.
* get_successors function: Generates the successors of a given state.
* astar function: Performs the A* search algorithm.
* bfs function: Performs the Breadth-First Search algorithm.
* dfs function: Performs the Depth-First Search algorithm.
* ids function: Performs the Iterative Deepening Search algorithm.
* greedy function: Performs the Greedy Best-First Search algorithm.
* ucs function: Performs the Uniform-Cost Search algorithm.
* dls function: Performs the Depth-Limited Search algorithm.
* main function: Parses the command-line arguments, reads the start and goal states from the input files, and calls the appropriate search algorithm.
* Supported Search Algorithms:
The program supports the following search algorithms:
* Breadth-First Search (bfs)
* Depth-First Search (dfs)
* Depth-Limited Search (dls)
* Uniform-Cost Search (ucs)
* Iterative Deepening Search (ids)
* A* Search (astar)
* Greedy Search (greedy)

Requirements
OS: Windows or Mac
This program requires Python 3 to run.

Example
Here is an example of how to run the program:
python expense_8_puzzle.py start.txt goal.txt astar True

Note: If any of the output values is negative consider goal state is not reached.

This runs the A* search algorithm using the start state in start.txt, the goal state in goal.txt, and outputs the search results to a file.
