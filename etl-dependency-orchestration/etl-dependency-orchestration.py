def schedule_pipeline(tasks, resource_budget):
    task_map = {t["name"]: t for t in tasks}
    n = len(tasks)

    completed = set()
    running = {}  # name -> end_time
    scheduled = set()
    schedule = []

    current_time = 0

    while len(completed) < n:
        # Step 1: complete tasks whose end time has been reached
        finished = [name for name, end in running.items() if end <= current_time]
        for name in finished:
            completed.add(name)
            del running[name]

        # Step 2: identify ready tasks
        ready = [
            t for t in tasks
            if t["name"] not in completed
            and t["name"] not in running
            and t["name"] not in scheduled
            and all(dep in completed for dep in t["depends_on"])
        ]

        # Step 3: sort alphabetically
        ready.sort(key=lambda t: t["name"])

        # Step 4: greedily assign, respecting resource budget
        current_usage = sum(task_map[name]["resources"] for name in running)
        for t in ready:
            if current_usage + t["resources"] <= resource_budget:
                schedule.append((t["name"], current_time))
                running[t["name"]] = current_time + t["duration"]
                scheduled.add(t["name"])
                current_usage += t["resources"]

        if len(completed) == n:
            break

        # Step 5: advance time to next completion event
        if running:
            current_time = min(running.values())
        else:
            # No running tasks and not all completed: nothing left schedulable.
            # Shouldn't happen for a valid DAG with resources <= budget.
            break

    schedule.sort(key=lambda x: (x[1], x[0]))
    return schedule