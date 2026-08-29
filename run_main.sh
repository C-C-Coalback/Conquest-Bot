TARGET=10000

while true; do
    # is 0 if file does not exist
    CURRENT=$(cat "num_games_done_tracker.txt" 2>/dev/null || echo 0)

    if [ "$CURRENT" -ge "$TARGET" ]; then
        echo "FINSIHED LOOPING"
        break
    fi

    echo "Start loop no $CURRENT"
    python3 main.py

    EXIT_CODE=$?
    echo "exit code $EXIT_CODE"
    echo "if 0, websocket got closed"
    echo "if 1, big problem"
    echo "restarting regardless"
    sleep 5
done