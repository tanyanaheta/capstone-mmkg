#! /bin/bash
if [[ $1 = "" ]]
then
    printf "Please provide a slurm script: \n\n  './launch_job.sh launch.slurm'\n\n"
    exit 0
else
    SCRIPT=$1
    USER_JOB="$(echo ${USER}_${SCRIPT})"
    JOB_NAME=${JOB_NAME:-zillow_MMKG}
fi
mkdir -p logs
JOB_ID=$(sbatch --job-name=$JOB_NAME --export=INSTANCE_NAME=$JOB_NAME $SCRIPT | awk '{print $NF}')
echo "Submitted batch job $JOB_ID"
while true
do
    if [[ -f "logs/${JOB_NAME}_${JOB_ID}.err" ]]
    then
        if ! squeue -u $USER -o "%i" | grep "$JOB_ID"
        then
            echo
            echo "Error:"
            tail "logs/${JOB_NAME}_${JOB_ID}.err"
            exit 0
        fi
    fi
    # if squeue -u $USER -o "%i %T" | grep "$JOB_ID RUNNING"; then
    if squeue -u $USER -o "%i %T" | grep "$JOB_ID RUNNING"
    then
        echo 
        echo "Running on node: $(squeue -u $USER -o '%A %B' | grep $JOB_ID | awk '{print $NF}')"
        echo
        break
    fi
    for i in {1..3}
    do  
        printf "\r\033[KWaiting for resources"
        sleep 0.75
        printf "\r\033[KWaiting for resources."
        sleep 0.75
        printf "\r\033[KWaiting for resources.."
        sleep 0.75
        printf "\r\033[KWaiting for resources..."
        sleep 0.75
    done
done
sleep 5
tail -f "$(ls logs/* -1t | head -2 | grep '.err')" | sed 's/^/STDERR: /' &
tail -f "$(ls logs/* -1t | head -2 | grep '.out')" | sed 's/^/STDOUT: /' 