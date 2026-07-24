import asyncio
import json

async def execute_in_isaac(source: str, host="127.0.0.1", port=8226) -> dict:
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(source.encode())
    writer.write_eof()
    data = await reader.read()
    writer.close()
    return json.loads(data.decode())

with open("ur7e/client.py", "r") as f:
    code = f.read()

#robot_1.apply_action(ArticulationAction(joint_positions=np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]), joint_indices=[0,1,2,3,4,5]))

result = asyncio.run(execute_in_isaac(code))
print(result["output"])
if result.get("traceback"):
    print(result["traceback"][0])