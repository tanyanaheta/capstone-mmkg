run:
	@./launch_job.sh launch.slurm

build:
	@./scripts/layer_setup.sh

sing:
	@./scripts/start_singularity_instance.sh

getnode:
	@./launch_job.sh getnode.slurm

clean:
	@rm -r ./scripts/overlays

rebuild: clean build
