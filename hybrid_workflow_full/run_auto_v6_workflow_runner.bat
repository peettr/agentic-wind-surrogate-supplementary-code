@echo off
cd /d <USER_HOME>\.openclaw\workspace-auto-v6
<USER_HOME>\AppData\Local\Programs\Python\Python312\python.exe <HYBRID_SOURCE_ROOT>\run_auto_v6_workflow_runner_with_notify.py >> <HYBRID_SOURCE_ROOT>\auto_v6_task_stdout.log 2>&1
