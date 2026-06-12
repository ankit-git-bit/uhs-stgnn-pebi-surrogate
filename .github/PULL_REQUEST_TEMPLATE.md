# Pull Request Template

## Description
Please include a summary of the changes and the scientific/engineering rationale. Specify which issue or feature request this PR addresses.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature / enhancement (non-breaking change which adds functionality)
- [ ] Documentation update
- [ ] Refactoring / optimization

## Scientific Validation & Reproducibility
Please describe the validation checks performed to ensure that the modifications do not introduce silent physical or numerical discrepancies.
- [ ] Did you run the automated verification pipeline (`reproduce_results.sh` / `reproduce_results.ps1`)?
- [ ] Are existing GNN model checkpoints still loading and evaluating successfully?
- [ ] Have you verified that no scientific figures/metrics from the published paper are altered (unless explicitly intended)?

## Checklist
- [ ] My code follows the code style guidelines of this repository.
- [ ] I have performed a self-review of my own code.
- [ ] I have commented my code, particularly in hard-to-understand areas or physics formulations.
- [ ] I have updated the documentation (such as `README.md` or API references) accordingly.
- [ ] My changes generate no new warnings or lint errors.
