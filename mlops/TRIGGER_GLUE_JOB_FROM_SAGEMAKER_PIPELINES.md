## Trigger glue job from sagemaker pipelines step

flowchart TD```
    A[SageMaker Pipeline] --> B[CallbackStep]
    B --> C[Lambda: Trigger Glue Job]
    C --> D[Start Glue JobRun]
    D --> E[Glue Job Execution]

    E --> F[EventBridge: Job State Change]

    F --> G[Lambda: Glue Job Complete Handler]
    G --> H[DynamoDB: Lookup CallbackToken]

    H -->|Job Succeeded| I[SageMaker API: send_pipeline_execution_step_success]
    H -->|Job Failed| J[SageMaker API: send_pipeline_execution_step_failure]

    I --> K[Resume SageMaker Pipeline]
    J --> K[Resume SageMaker Pipeline]
    ```
