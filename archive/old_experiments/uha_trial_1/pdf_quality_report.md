# PDF Quality & LaTeX Audit Report

This report evaluates the monolithic LaTeX manuscript ([final_paper.tex](file:///c:/Users/user/Desktop/KTU%20intern/sciml%20models/final_paper.tex)) against academic typesetting standards, focusing on twocolumn formatting, overfull boxes, figure placement, and mathematical typography.

---

## 1. Summary of PDF Quality Audit

* **Typesetting Engine**: pdfLaTeX / XeLaTeX
* **Layout**: Standard article class, 10pt, twocolumn format, 0.75in margins.
* **Page Count Estimate**: Approximately 5.5 to 6 pages (including 12 figures and 5 tables).
* **Reference Style**: IEEE/IEEETR BibTeX style (standard numbered brackets).

---

## 2. Issues Found & Suggested Fixes

### Issue 1: Improper Placement Specifiers for Double-Column Figures (`figure*`)
* **Severity**: **High**
* **Location**: `fig:dataset_overview`, `fig:pressure_evolution`, `fig:pressure_tracking`, `fig:saturation_tracking`
* **Problem**: These figures are wrapped in `\begin{figure*}` to span across both columns. However, they use the `[htbp]` placement specifier. In standard LaTeX double-column layouts, `figure*` environments do **not** support bottom `[b]` or here `[h]` placement. Using `[htbp]` causes LaTeX to default to placing them on a separate page at the very end of the document, disrupting reading flow.
* **Suggested Fix**: Change the specifier to `[t]` or `[tp]` (top or top-page only) for all `figure*` environments:
  ```latex
  \begin{figure*}[t]
  ...
  \end{figure*}
  ```

### Issue 2: Potential Overfull `\hbox` from Long Equations in Twocolumn Mode
* **Severity**: **Medium-High**
* **Location**: Eq. 1 (mass conservation) and Eq. 4 (transmissibility GCN aggregation)
* **Problem**: In a standard 10pt twocolumn format, the column width is approximately 3.25 inches (8.25 cm). Equation 1 and Equation 4 contain large fractions, vector operators, and subscript terms that span too wide. Compiling these equations will likely trigger overfull `\hbox` warnings, causing the equation numbers to overlap with the formulas or clip off the page margins.
* **Suggested Fix**: Use the `split` or `aligned` environment inside the equation block to break the formulas across multiple lines:
  ```latex
  \begin{equation}
  \label{eq:mass_cons}
  \begin{aligned}
  \nabla \cdot \left[ \frac{\mathbf{k} k_{r\beta}}{\mu_\beta B_\beta} (\nabla P_\beta - \rho_\beta g \nabla Z) \right] &+ q_\beta \\
  &= \frac{\partial}{\partial t} \left( \frac{\phi S_\beta}{B_\beta} \right)
  \end{aligned}
  \end{equation}
  ```

### Issue 3: Table Column Overflow in Table 1 (`dataset_stats`)
* **Severity**: **Medium**
* **Location**: `tab:dataset_stats`
* **Problem**: The column specifiers are `llcc`. The second column description (`Physical Meaning`) contains entries like "Absolute Permeability (mD)" and "Injection H₂ Rate (daily rate units)". In a narrow column layout, this long text will stretch the table beyond the 3.25-inch column boundary, clipping the mean and standard deviation columns.
* **Suggested Fix**: Use the `tabularx` package or wrap the column text inside a fixed-width column descriptor `p{2.5cm}` or `p{3.0cm}` to force line wrapping, or shorten the description texts (e.g. "Abs. Permeability (mD)", "H₂ Injection Rate (vol/d)").
  ```latex
  \begin{tabular}{lp{3.2cm}cc}
  ```

### Issue 4: Text-mode Underscores in Inline Code
* **Severity**: **Medium**
* **Location**: Section 6 (`time\_since\_inj\_start`)
* **Problem**: Although the underscores are properly escaped (`\_`), typing code variable names in standard text font makes them hard to read and disrupts scientific flow. It also causes bad line breaking.
* **Suggested Fix**: Wrap all code-level variable names and feature flags in the typewriter font `\texttt{}`:
  ```latex
  \texttt{time\_since\_inj\_start}
  ```

### Issue 5: Raster Graphics vs. Vector Graphics Standard
* **Severity**: **Low-Medium**
* **Location**: All 12 figures
* **Problem**: All figures are embedded as rasterized `.png` files. While acceptable for initial submissions, many computational geosciences journals and SPE/EAGE conferences require vector graphics (`.pdf` or `.eps`) for line plots, flowcharts, and diagrams (such as the loss curves and tracking plots) to prevent pixelation during high-resolution printing.
* **Suggested Fix**: Export the line plots (Fig. 1, 4, 8, 9, 10, 11, 12) from Matplotlib as `.pdf` vectors, and only retain high-resolution `.png` files for 3D spatial field snapshots (Fig. 3, 6).

### Issue 6: Table Placement and Floating
* **Severity**: **Low**
* **Location**: Table 1, Table 2, Table 3, Table 4, Table 5
* **Problem**: All tables use the strict `[h]` or `[h!]` placement. If a table cannot fit on the current page column, it will block subsequent text rendering or create large blank areas at the bottom of the page.
* **Suggested Fix**: Relax the placement specifiers to `[htbp]` to allow LaTeX's layout engine to find the optimal page position.
