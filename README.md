# mlPrediction

for debugging flask in vscode you can use:

"version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Flask",
            "type": "python",
            "request": "launch",
            "module": "flask",
            "env": {
                "FLASK_APP": "mlPrediction.py",
                "FLASK_ENV": "development",
                "FLASK_DEBUG": "0"

            },
            "args": [
                "run",
                "--no-debugger",
                "--no-reload"
            ],
            "jinja": true
        }
    ]