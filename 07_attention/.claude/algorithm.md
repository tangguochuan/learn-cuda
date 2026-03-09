**Algorithm 2** FlashAttention-2 Backward Pass

---

**Require:** Matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}, d\mathbf{O} \in \mathbb{R}^{N \times d}$ in HBM, vector $\mathbf{L} \in \mathbb{R}^N$ in HBM, block sizes $B_c, B_r$.

1. Divide $\mathbf{Q}$ into $T_r = \left\lceil \frac{N}{B_r} \right\rceil$ blocks $\mathbf{Q}_1, \ldots, \mathbf{Q}_{T_r}$ of size $B_r \times d$ each, and divide $\mathbf{K}, \mathbf{V}$ in to $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ blocks $\mathbf{K}_1, \ldots, \mathbf{K}_{T_c}$ and $\mathbf{V}_1, \ldots, \mathbf{V}_{T_c}$, of size $B_c \times d$ each.

2. Divide $\mathbf{O}$ into $T_r$ blocks $\mathbf{O}_i, \ldots, \mathbf{O}_{T_r}$ of size $B_r \times d$ each, divide $d\mathbf{O}$ into $T_r$ blocks $d\mathbf{O}_i, \ldots, d\mathbf{O}_{T_r}$ of size $B_r \times d$ each, and divide $\mathbf{L}$ into $T_r$ blocks $\mathbf{L}_i, \ldots, \mathbf{L}_{T_r}$ of size $B_r$ each.

3. Initialize $d\mathbf{Q} = (0)_{N \times d}$ in HBM and divide it into $T_r$ blocks $d\mathbf{Q}_1, \ldots, d\mathbf{Q}_{T_r}$ of size $B_r \times d$ each. Divide $d\mathbf{K}, d\mathbf{V} \in \mathbb{R}^{N \times d}$ in to $T_c$ blocks $d\mathbf{K}_1, \ldots, d\mathbf{K}_{T_c}$ and $d\mathbf{V}_1, \ldots, d\mathbf{V}_{T_c}$, of size $B_c \times d$ each.

4. Compute $\mathbf{D} = \text{rowsum}(d\mathbf{O} \circ \mathbf{O}) \in \mathbb{R}^d$ (pointwise multiply), write $\mathbf{D}$ to HBM and divide it into $T_r$ blocks $\mathbf{D}_1, \ldots, \mathbf{D}_{T_r}$ of size $B_r$ each.

5. **for** $1 \leq j \leq T_c$ **do**

6. $\quad$ Load $\mathbf{K}_j, \mathbf{V}_j$ from HBM to on-chip SRAM.

7. $\quad$ Initialize $d\mathbf{K}_j = (0)_{B_c \times d}, d\mathbf{V}_j = (0)_{B_c \times d}$ on SRAM.

8. $\quad$ **for** $1 \leq i \leq T_r$ **do**

9. $\quad\quad$ Load $\mathbf{Q}_i, \mathbf{O}_i, d\mathbf{O}_i, d\mathbf{Q}_i, \mathbf{L}_i, \mathbf{D}_i$ from HBM to on-chip SRAM.

10. $\quad\quad$ On chip, compute $\mathbf{S}_i^{(j)} = \mathbf{Q}_i \mathbf{K}_j^T \in \mathbb{R}^{B_r \times B_c}$.

11. $\quad\quad$ On chip, compute $\mathbf{P}_i^{(j)} = \exp(\mathbf{S}_{ij} - \mathbf{L}_i) \in \mathbb{R}^{B_r \times B_c}$.

12. $\quad\quad$ On chip, compute $d\mathbf{V}_j \leftarrow d\mathbf{V}_j + (\mathbf{P}_i^{(j)})^T d\mathbf{O}_i \in \mathbb{R}^{B_c \times d}$.

13. $\quad\quad$ On chip, compute $d\mathbf{P}_i^{(j)} = d\mathbf{O}_i \mathbf{V}_j^T \in \mathbb{R}^{B_r \times B_c}$.

14. $\quad\quad$ On chip, compute $d\mathbf{S}_i^{(j)} = \mathbf{P}_i^{(j)} \circ (d\mathbf{P}_i^{(j)} - \mathbf{D}_i) \in \mathbb{R}^{B_r \times B_c}$.

15. $\quad\quad$ Load $d\mathbf{Q}_i$ from HBM to SRAM, then on chip, update $d\mathbf{Q}_i \leftarrow d\mathbf{Q}_i + d\mathbf{S}_i^{(j)} \mathbf{K}_j \in \mathbb{R}^{B_r \times d}$, and write back to HBM.

16. $\quad\quad$ On chip, compute $d\mathbf{K}_j \leftarrow d\mathbf{K}_j + d\mathbf{S}_i^{(j)T} \mathbf{Q}_i \in \mathbb{R}^{B_c \times d}$.

17. $\quad$ **end for**

18. $\quad$ Write $d\mathbf{K}_j, d\mathbf{V}_j$ to HBM.

19. **end for**

20. **Return** $d\mathbf{Q}, d\mathbf{K}, d\mathbf{V}$.