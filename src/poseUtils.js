import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';

// Load the PoseNet model
export async function loadPoseNet() {
  await tf.ready();
  const net = await posenet.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    inputResolution: { width: 257, height: 257 },
    multiplier: 0.75
  });
  return net;
}

// Estimate pose from an image element
export async function estimatePose(net, imageElement) {
  if (!net || !imageElement) return null;
  try {
    const pose = await net.estimateSinglePose(imageElement, {
      flipHorizontal: false
    });
    return pose;
  } catch (error) {
    console.error("Error estimating pose:", error);
    return null;
  }
}

// Helper function to find a keypoint by name
const getKeypoint = (pose, part) => pose.keypoints.find(k => k.part === part);

// New helper function to calculate the angle between three points
function calculateAngle(p1, p2, p3) {
  const angle = Math.atan2(p3.y - p2.y, p3.x - p2.x) - Math.atan2(p1.y - p2.y, p1.x - p2.x);
  let angleDegrees = Math.abs(angle * (180 / Math.PI));
  if (angleDegrees > 180) {
    angleDegrees = 360 - angleDegrees;
  }
  return angleDegrees;
}

// Normalize pose keypoints based on a reference distance (e.g., shoulder distance)
function normalizePose(pose) {
  const leftShoulder = getKeypoint(pose, 'leftShoulder');
  const rightShoulder = getKeypoint(pose, 'rightShoulder');
  
  if (!leftShoulder || !rightShoulder) {
    return null;
  }

  const shoulderDist = Math.sqrt(
    Math.pow(leftShoulder.position.x - rightShoulder.position.x, 2) +
    Math.pow(leftShoulder.position.y - rightShoulder.position.y, 2)
  );
  
  if (shoulderDist === 0) return null;

  const normalizedKeypoints = pose.keypoints.map(k => ({
    ...k,
    position: {
      x: (k.position.x - leftShoulder.position.x) / shoulderDist,
      y: (k.position.y - leftShoulder.position.y) / shoulderDist,
    }
  }));

  return { ...pose, keypoints: normalizedKeypoints };
}


// Compare two poses with weighted keypoints
export function comparePoses(pose1, pose2) {
  if (!pose1 || !pose2) return false;

  const normalizedPose1 = normalizePose(pose1);
  const normalizedPose2 = normalizePose(pose2);

  if (!normalizedPose1 || !normalizedPose2) return false;

  const keypointWeights = {
    nose: 1,
    leftEye: 1,
    rightEye: 1,
    leftEar: 1,
    rightEar: 1,
    leftShoulder: 2,
    rightShoulder: 2,
    leftElbow: 3,
    rightElbow: 3,
    leftWrist: 4,
    rightWrist: 4,
    leftHip: 1,
    rightHip: 1,
    leftKnee: 1,
    rightKnee: 1,
    leftAnkle: 1,
    rightAnkle: 1
  };

  let totalWeightedDistance = 0;
  let totalWeight = 0;

  for (const part in keypointWeights) {
    const p1 = getKeypoint(normalizedPose1, part);
    const p2 = getKeypoint(normalizedPose2, part);
    
    if (p1 && p2 && p1.score > 0.5 && p2.score > 0.5) {
      const dx = p1.position.x - p2.position.x;
      const dy = p1.position.y - p2.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      const weight = keypointWeights[part];
      totalWeightedDistance += dist * weight;
      totalWeight += weight;
    }
  }

  if (totalWeight === 0) return false;

  const weightedAverageDistance = totalWeightedDistance / totalWeight;
  
  // Angle-based comparison
  const jointAngleComparisons = {
    leftElbow: [getKeypoint(normalizedPose1, 'leftShoulder'), getKeypoint(normalizedPose1, 'leftElbow'), getKeypoint(normalizedPose1, 'leftWrist')],
    rightElbow: [getKeypoint(normalizedPose1, 'rightShoulder'), getKeypoint(normalizedPose1, 'rightElbow'), getKeypoint(normalizedPose1, 'rightWrist')],
    leftShoulder: [getKeypoint(normalizedPose1, 'leftElbow'), getKeypoint(normalizedPose1, 'leftShoulder'), getKeypoint(normalizedPose1, 'leftHip')],
    rightShoulder: [getKeypoint(normalizedPose1, 'rightElbow'), getKeypoint(normalizedPose1, 'rightShoulder'), getKeypoint(normalizedPose1, 'rightHip')],
  };

  const jointAngleComparisons2 = {
    leftElbow: [getKeypoint(normalizedPose2, 'leftShoulder'), getKeypoint(normalizedPose2, 'leftElbow'), getKeypoint(normalizedPose2, 'leftWrist')],
    rightElbow: [getKeypoint(normalizedPose2, 'rightShoulder'), getKeypoint(normalizedPose2, 'rightElbow'), getKeypoint(normalizedPose2, 'rightWrist')],
    leftShoulder: [getKeypoint(normalizedPose2, 'leftElbow'), getKeypoint(normalizedPose2, 'leftShoulder'), getKeypoint(normalizedPose2, 'leftHip')],
    rightShoulder: [getKeypoint(normalizedPose2, 'rightElbow'), getKeypoint(normalizedPose2, 'rightShoulder'), getKeypoint(normalizedPose2, 'rightHip')],
  };

  let totalAngleDifference = 0;
  let anglesCompared = 0;

  for (const joint in jointAngleComparisons) {
    const points1 = jointAngleComparisons[joint];
    const points2 = jointAngleComparisons2[joint];

    if (points1.every(p => p && p.score > 0.5) && points2.every(p => p && p.score > 0.5)) {
      const angle1 = calculateAngle(points1[0].position, points1[1].position, points1[2].position);
      const angle2 = calculateAngle(points2[0].position, points2[1].position, points2[2].position);
      
      totalAngleDifference += Math.abs(angle1 - angle2);
      anglesCompared++;
    }
  }

  const averageAngleDifference = anglesCompared > 0 ? totalAngleDifference / anglesCompared : 0;
  
  console.log("Weighted average pose distance:", weightedAverageDistance);
  console.log("Average angle difference (degrees):", averageAngleDifference);

  // Combined threshold logic
  const distanceThreshold = 0.4;
  const angleThreshold = 12;

  const posesAreSimilar = weightedAverageDistance < distanceThreshold && averageAngleDifference < angleThreshold;

  return posesAreSimilar;
}

