.PHONY: validate smoke clean

validate:
	python3 -m compileall -q src
	bash -n slurm/qualification.sbatch slurm/fabric.sbatch slurm/admission.sbatch slurm/submit_qualification.sh examples/ib_write_bw_pair.sh
	PYTHONPATH=src python3 -m clustercheck --version

smoke:
	rm -rf /tmp/clustercheck-smoke /tmp/clustercheck-fixtures
	PYTHONPATH=src python3 -m clustercheck simulate --config configs/synthetic-demo.toml --output /tmp/clustercheck-smoke --run-id smoke --baseline-fixture-output /tmp/clustercheck-fixtures || test $$? -eq 2

clean:
	rm -rf build dist src/clustercheck.egg-info .mypy_cache .ruff_cache
