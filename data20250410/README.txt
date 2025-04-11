set of short-time RK4 time evolution (QTT code, but no compression)

initialization: 
electron distribution function: superposition of cosines

        for nx, nv in np.ndindex(mx, mv):
            out = out + np.cos(2 * np.pi * (nx * x_mesh + nv * v_mesh + np.random.random(1)))

three folders of test files:
- mx=8, mv=8
- mx=8, mv=4
- mx=4, mv=8

initial state and final time evolved state are stored in
'fe_seed{seed}_k0.1_mr25.0_C20_mbNone_TNPGFF_Ls{L:02d},{L:02d},{L:02d}_o1_cfl0.9_T1.0_te4'

seed = range(100)
L = 7 or 8
