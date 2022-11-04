run:
	@./launch_job.sh launch.slurm

build:
	@./layer_setup.sh

getnode:
	@./launch_job.sh getnode.slurm

clean:
	@rm overlay-*

rebuild: clean build