import numpy as np

np.set_printoptions(precision=4, suppress=True)  # for clean output

# -----------------------------------------------------
# Step 1: Create a simple input sample (3 sites × 2 features)
# -----------------------------------------------------
x = np.array([
    [1.0, 2.0],   # x[0]
    [3.0, 4.0],   # x[1]
    [5.0, 6.0]    # x[2]
])

print("🟦 Input Sample x:")
print(x)
print("Shape:", x.shape)  # (3, 2)

# -----------------------------------------------------
# Step 2: Create MPS Tensors
# -----------------------------------------------------

# MPS[0] shape: (2, 2)
MPS_0 = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

# MPS[1] shape: (2, 2, 2)
MPS_1 = np.array([
    [[0.1, 0.2],
     [0.3, 0.4]],
     
    [[0.5, 0.6],
     [0.7, 0.8]]
])

# MPS[2] shape: (2, 2)
MPS_2 = np.array([
    [0.9, 1.0],
    [1.1, 1.2]
])

print("\n🧱 MPS Tensors:")
print("MPS_0 shape:", MPS_0.shape)
print("MPS_1 shape:", MPS_1.shape)
print("MPS_2 shape:", MPS_2.shape)

# -----------------------------------------------------
# Step 3: First contraction → x[0] × MPS_0
# -----------------------------------------------------
result = np.tensordot(x[0], MPS_0, axes=[0, 0])
# x[0]: shape (2,)
# MPS_0: shape (2, 2)
# result: shape (2,)

print("\n🔁 Step 3: Contract x[0] with MPS_0")
print("x[0]:", x[0])
print("result =", result)
print("Shape:", result.shape)

# Manual math:
# result[0] = 1.0*0.1 + 2.0*0.3 = 0.1 + 0.6 = 0.7
# result[1] = 1.0*0.2 + 2.0*0.4 = 0.2 + 0.8 = 1.0

# -----------------------------------------------------
# Step 4: Middle contraction → MPS_1 × x[1], then × result
# -----------------------------------------------------
temp = np.tensordot(MPS_1, x[1], axes=[[1], [0]])
# MPS_1: (2, 2, 2), x[1]: (2,) → temp: (2, 2)

print("\n🔁 Step 4a: Contract MPS_1 with x[1]")
print("x[1]:", x[1])
print("temp =\n", temp)
print("Shape:", temp.shape)

# Now contract `result` (from step 3) with `temp`
result = np.tensordot(result, temp, axes=[0, 0])
# result: (2,), temp: (2, 2) → result: (2,)

print("\n🔁 Step 4b: Contract previous result with temp")
print("New result =", result)
print("Shape:", result.shape)

# -----------------------------------------------------
# Step 5: Final contraction → MPS_2 × x[2], then × result
# -----------------------------------------------------
final_tensor = np.tensordot(MPS_2, x[2], axes=[1, 0])
# MPS_2: (2, 2), x[2]: (2,) → final_tensor: (2,)

print("\n🔁 Step 5a: Contract MPS_2 with x[2]")
print("x[2]:", x[2])
print("final_tensor =", final_tensor)
print("Shape:", final_tensor.shape)

# Now final contraction
final_output = np.tensordot(result, final_tensor, axes=[0, 0])
# Both (2,) → scalar

print("\n✅ Final Step: Contract result with final_tensor")
print("Final Output =", final_output)
print("Shape:", final_output.shape)
