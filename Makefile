run:
	@./launch_job.sh launch.slurm

build:
	@./scripts/layer_setup.sh

getnode:
	@./launch_job.sh getnode.slurm

clean:
	@rm /scripts/overlays/overlay-*

rebuild: clean build
