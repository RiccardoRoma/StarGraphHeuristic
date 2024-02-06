import generate_star_states
edges_list = [(0,2), (1,3), (1,5), (2,5), (2,7), (2,9), (3,6), (3,8), (4,5), (5,9)]
center_list = [2, 5, 3]

c = generate_star_states.generate_star_states(edges_list, center_list, add_barries=True)
print(c.draw())

