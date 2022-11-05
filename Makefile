env:
	@export PROJECT_ROOT=${PWD}
	
run: env
	@./launch_job.sh launch.slurm

build: env
	@./scripts/layer_setup.sh

getnode: env
	@./launch_job.sh getnode.slurm

clean:
	@rm -f ./overlays/overlay-*

rebuild: clean build
