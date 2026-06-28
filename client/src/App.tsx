import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { ASPECT_RATIO, PINCH_THRESHOLD } from '../../shared/const';

type Point = { x: number; y: number };

// ==========================================
// Shader Programs
// ==========================================

const VERTEX_SHADER = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAGMENT_SHADER = `
uniform sampler2D tDiffuse;
uniform vec2 resolution;
uniform float time;
uniform float uMode;

varying vec2 vUv;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

void main() {
  // Mirror texture coordinates horizontally
  vec2 texCoord = vec2(1.0 - vUv.x, vUv.y);
  vec4 videoColor = texture2D(tDiffuse, texCoord);
  float brightness = dot(videoColor.rgb, vec3(0.299, 0.587, 0.114));

  // ------------------------------------------
  // Mode 0: Particle Effect
  // ------------------------------------------
  float bands = 15.0;
  float bandValue = fract(brightness * bands - time * 0.5);
  float contour = smoothstep(0.46, 0.5, bandValue) * (1.0 - smoothstep(0.5, 0.54, bandValue));
  vec3 contourColor = vec3(0.05, 0.3, 0.9) * 1.5 * contour;

  vec2 gridRes = vec2(90.0);
  vec2 cellIndex = floor(vUv * gridRes);
  vec2 cellCenter = (cellIndex + 0.5) / gridRes;
  vec2 cellTexCoord = vec2(1.0 - cellCenter.x, cellCenter.y);
  vec4 cellVideoColor = texture2D(tDiffuse, cellTexCoord);
  float cellBrightness = dot(cellVideoColor.rgb, vec3(0.299, 0.587, 0.114));

  vec2 localUv = fract(vUv * gridRes);
  float cellPhase = cellIndex.x * 12.9898 + cellIndex.y * 78.233;
  float twinkle = sin(time * 12.0 + cellPhase) * 0.4 + 0.6;

  float colorCycle = time * 15.0 + cellPhase;
  float colorIndex = mod(floor(colorCycle), 3.0);
  float colorFract = fract(colorCycle);
  vec3 colA, colB;
  if (colorIndex == 0.0) {
    colA = vec3(1.0, 0.2, 0.6); // magenta
    colB = vec3(1.0, 0.9, 0.2); // yellow
  } else if (colorIndex == 1.0) {
    colA = vec3(1.0, 0.9, 0.2); // yellow
    colB = vec3(1.0, 1.0, 1.0); // white
  } else {
    colA = vec3(1.0, 1.0, 1.0); // white
    colB = vec3(1.0, 0.2, 0.6); // magenta
  }
  vec3 cycledColor = mix(colA, colB, colorFract);

  float sizeThreshold = cellBrightness * 0.7;
  float inParticle = step(0.5 - sizeThreshold * 0.5, localUv.x) * step(localUv.x, 0.5 + sizeThreshold * 0.5) *
                     step(0.5 - sizeThreshold * 0.5, localUv.y) * step(localUv.y, 0.5 + sizeThreshold * 0.5);

  vec3 particleGridColor = cycledColor * inParticle * twinkle;
  vec3 subjectGlow = vec3(0.1, 0.3, 0.9) * smoothstep(0.2, 0.8, brightness) * 0.6;
  vec3 particleFinalColor = contourColor + particleGridColor + subjectGlow;

  // ------------------------------------------
  // Mode 1: X-Ray Effect
  // ------------------------------------------
  vec3 lightBlue = vec3(0.02, 0.25, 0.7);
  vec3 darkBlue = vec3(0.0, 0.02, 0.15);
  float faceMask = smoothstep(0.15, 0.75, brightness);
  vec3 xrayBaseColor = mix(lightBlue, darkBlue, faceMask);

  vec2 texelSize = 1.0 / resolution;
  float m00 = dot(texture2D(tDiffuse, vec2(1.0 - (vUv.x - texelSize.x), vUv.y - texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
  float m01 = dot(texture2D(tDiffuse, vec2(1.0 - vUv.x,                 vUv.y - texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
  float m02 = dot(texture2D(tDiffuse, vec2(1.0 - (vUv.x + texelSize.x), vUv.y - texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
  
  float m10 = dot(texture2D(tDiffuse, vec2(1.0 - (vUv.x - texelSize.x), vUv.y)).rgb, vec3(0.299, 0.587, 0.114));
  float m12 = dot(texture2D(tDiffuse, vec2(1.0 - (vUv.x + texelSize.x), vUv.y)).rgb, vec3(0.299, 0.587, 0.114));
  
  float m20 = dot(texture2D(tDiffuse, vec2(1.0 - (vUv.x - texelSize.x), vUv.y + texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
  float m21 = dot(texture2D(tDiffuse, vec2(1.0 - vUv.x,                 vUv.y + texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
  float m22 = dot(texture2D(tDiffuse, vec2(1.0 - (vUv.x + texelSize.x), vUv.y + texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));

  float edgeX = (m02 + 2.0 * m12 + m22) - (m00 + 2.0 * m10 + m20);
  float edgeY = (m20 + 2.0 * m21 + m22) - (m00 + 2.0 * m01 + m02);
  float edgeMag = sqrt(edgeX * edgeX + edgeY * edgeY);
  vec3 cyanEdges = vec3(0.0, 0.9, 1.0) * edgeMag * 2.0;

  float noiseVal = hash(vUv + time * 100.0) * 0.1 - 0.05;
  float scanline = sin(vUv.y * resolution.y * 2.0) * 0.05;

  vec3 xrayFinalColor = xrayBaseColor + cyanEdges + vec3(noiseVal) - vec3(scanline);

  // ------------------------------------------
  // Mode Blending
  // ------------------------------------------
  vec3 finalColor = mix(particleFinalColor, xrayFinalColor, uMode);
  gl_FragColor = vec4(finalColor, 0.95);
}
`;

// ==========================================
// XRayWindow Component
// ==========================================

interface XRayWindowProps {
  pointsRef: React.RefObject<Point[]>;
  videoTexture: THREE.VideoTexture | null;
  videoWidth: number;
  videoHeight: number;
  effectMode: 'particle' | 'xray';
}

const XRayWindow: React.FC<XRayWindowProps> = ({
  pointsRef,
  videoTexture,
  videoWidth,
  videoHeight,
  effectMode
}) => {
  const meshRef = useRef<THREE.Mesh>(null);

  const uniforms = useMemo(() => ({
    tDiffuse: { value: null as THREE.VideoTexture | null },
    resolution: { value: new THREE.Vector2(videoWidth, videoHeight) },
    time: { value: 0.0 },
    uMode: { value: 0.0 }
  }), [videoWidth, videoHeight]);

  useEffect(() => {
    uniforms.tDiffuse.value = videoTexture;
  }, [videoTexture, uniforms]);

  useFrame((state) => {
    const { clock } = state;
    const time = clock.getElapsedTime();
    uniforms.time.value = time;

    const targetMode = effectMode === 'xray' ? 1.0 : 0.0;
    uniforms.uMode.value += (targetMode - uniforms.uMode.value) * 0.15;

    const mesh = meshRef.current;
    if (mesh && pointsRef.current && pointsRef.current.length === 4) {
      const geometry = mesh.geometry as THREE.PlaneGeometry;
      const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
      const uvAttr = geometry.getAttribute('uv') as THREE.BufferAttribute;

      const points = pointsRef.current;
      const count = posAttr.count;

      for (let i = 0; i < count; i++) {
        const u = uvAttr.getX(i);
        const v = uvAttr.getY(i);

        // Bilinear interpolation formula:
        // P(u, v) = (1 - u)*(1 - v)*BL + u*(1 - v)*BR + (1 - u)*v*TL + u*v*TR
        // Where:
        // points[0] = TL, points[1] = TR, points[2] = BL, points[3] = BR
        const ptX = (1.0 - u) * (1.0 - v) * points[2].x +
                    u * (1.0 - v) * points[3].x +
                    (1.0 - u) * v * points[0].x +
                    u * v * points[1].x;

        const ptY = (1.0 - u) * (1.0 - v) * points[2].y +
                    u * (1.0 - v) * points[3].y +
                    (1.0 - u) * v * points[0].y +
                    u * v * points[1].y;

        const x_3d = (1.0 - ptX) * 2.0 - 1.0;
        const y_3d = (1.0 - ptY) * 2.0 - 1.0;

        posAttr.setXY(i, x_3d, y_3d);
        uvAttr.setXY(i, 1.0 - ptX, 1.0 - ptY);
      }

      posAttr.needsUpdate = true;
      uvAttr.needsUpdate = true;
    }
  });

  return (
    <mesh ref={meshRef}>
      <planeGeometry args={[2, 2, 32, 32]} />
      <shaderMaterial
        vertexShader={VERTEX_SHADER}
        fragmentShader={FRAGMENT_SHADER}
        uniforms={uniforms}
        transparent={true}
        side={THREE.DoubleSide}
        depthWrite={false}
        depthTest={false}
      />
    </mesh>
  );
};

// ==========================================
// Main App Component
// ==========================================

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const bgVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);

  const pointsRef = useRef<Point[]>([]);
  const effectModeRef = useRef<'particle' | 'xray'>('particle');
  const requestRef = useRef<number>(-1);

  const [pointsState, setPointsState] = useState<Point[]>([]);
  const [effectMode, setEffectMode] = useState<'particle' | 'xray'>('particle');
  const [isReady, setIsReady] = useState(false);
  const [videoWidth, setVideoWidth] = useState(1280);
  const [videoHeight, setVideoHeight] = useState(720);
  const [videoAspect, setVideoAspect] = useState(ASPECT_RATIO);
  const [videoTexture, setVideoTexture] = useState<THREE.VideoTexture | null>(null);
  const [containerStyle, setContainerStyle] = useState<React.CSSProperties>({ width: '100%', height: '100%' });

  // Handle container resizing to lock aspect ratio
  useEffect(() => {
    const handleResize = () => {
      const windowAspect = window.innerWidth / window.innerHeight;
      if (windowAspect > videoAspect) {
        setContainerStyle({ width: `${window.innerHeight * videoAspect}px`, height: '100%' });
      } else {
        setContainerStyle({ width: '100%', height: `${window.innerWidth / videoAspect}px` });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [videoAspect]);

  // MediaPipe and Camera initialization
  useEffect(() => {
    let active = true;
    let stream: MediaStream | null = null;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });

        if (!active) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;

        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user"
          },
          audio: false
        });

        if (!active) {
          stream.getTracks().forEach(t => t.stop());
          return;
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(err => console.warn("Video play error:", err));
        }

        if (bgVideoRef.current) {
          bgVideoRef.current.srcObject = stream;
          await bgVideoRef.current.play().catch(err => console.warn("Background video play error:", err));
        }

        if (bgVideoRef.current) {
          const texture = new THREE.VideoTexture(bgVideoRef.current);
          texture.colorSpace = THREE.SRGBColorSpace;
          setVideoTexture(texture);
        }

        setIsReady(true);
      } catch (err) {
        console.error("Initialization failed:", err);
      }
    }

    init();

    return () => {
      active = false;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
      }
      if (requestRef.current !== -1) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, []);

  // Frame-by-frame hand detection loop
  useEffect(() => {
    if (!isReady || !landmarkerRef.current) return;

    const detectHands = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video && video.readyState >= 2 && landmarkerRef.current && canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          const timestamp = performance.now();
          const results = landmarkerRef.current.detectForVideo(video, timestamp);

          // Draw skeleton joints and bones
          if (results.landmarks && results.landmarks.length > 0) {
            ctx.save();
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.lineWidth = 1;
            ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
            ctx.shadowBlur = 3;

            results.landmarks.forEach((handLandmarks) => {
              CONNECTIONS.forEach(([i, j]) => {
                const pt1 = handLandmarks[i];
                const pt2 = handLandmarks[j];
                if (pt1 && pt2) {
                  ctx.beginPath();
                  ctx.moveTo((1.0 - pt1.x) * canvas.width, pt1.y * canvas.height);
                  ctx.lineTo((1.0 - pt2.x) * canvas.width, pt2.y * canvas.height);
                  ctx.stroke();
                }
              });

              handLandmarks.forEach((pt) => {
                ctx.beginPath();
                ctx.arc((1.0 - pt.x) * canvas.width, pt.y * canvas.height, 1.5, 0, 2 * Math.PI);
                ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.fill();
              });
            });

            ctx.restore();
          }

          // Evaluate pinch and window geometry
          if (results.landmarks && results.landmarks.length >= 2) {
            const sortedHands = [...results.landmarks].sort((h1, h2) => {
              const x1 = 1.0 - h1[8].x;
              const x2 = 1.0 - h2[8].x;
              return x1 - x2;
            });

            const handL = sortedHands[0];
            const handR = sortedHands[1];

            const isPinching = (hand: any) => {
              const dx = hand[8].x - hand[4].x;
              const dy = hand[8].y - hand[4].y;
              return Math.sqrt(dx * dx + dy * dy) < PINCH_THRESHOLD;
            };

            const pinchL = isPinching(handL);
            const pinchR = isPinching(handR);

            if (pinchL && pinchR) {
              const pinchCenterL = {
                x: (handL[8].x + handL[4].x) / 2.0,
                y: (handL[8].y + handL[4].y) / 2.0
              };
              const pinchCenterR = {
                x: (handR[8].x + handR[4].x) / 2.0,
                y: (handR[8].y + handR[4].y) / 2.0
              };

              const pts = [
                { x: pinchCenterL.x, y: pinchCenterL.y - 0.075 }, // TL
                { x: pinchCenterR.x, y: pinchCenterR.y - 0.075 }, // TR
                { x: pinchCenterL.x, y: pinchCenterL.y + 0.075 }, // BL
                { x: pinchCenterR.x, y: pinchCenterR.y + 0.075 }  // BR
              ];

              setPointsState(pts);
              pointsRef.current = pts;
              setEffectMode('xray');
              effectModeRef.current = 'xray';
            } else {
              const pts = [
                { x: handL[8].x, y: handL[8].y }, // TL
                { x: handR[8].x, y: handR[8].y }, // TR
                { x: handL[4].x, y: handL[4].y }, // BL
                { x: handR[4].x, y: handR[4].y }  // BR
              ];

              setPointsState(pts);
              pointsRef.current = pts;
              setEffectMode('particle');
              effectModeRef.current = 'particle';
            }
          } else {
            setPointsState([]);
            pointsRef.current = [];
          }
        }
      }
      requestRef.current = requestAnimationFrame(detectHands);
    };

    requestRef.current = requestAnimationFrame(detectHands);

    return () => {
      if (requestRef.current !== -1) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [isReady]);

  const handleVideoMetadata = () => {
    if (videoRef.current) {
      const w = videoRef.current.videoWidth;
      const h = videoRef.current.videoHeight;
      setVideoWidth(w);
      setVideoHeight(h);
      setVideoAspect(w / h);
    }
  };

  return (
    <div className="w-screen h-screen bg-black flex items-center justify-center overflow-hidden">
      <video
        ref={videoRef}
        style={{ opacity: 0, pointerEvents: 'none', position: 'absolute', width: '1px', height: '1px' }}
        playsInline
        muted
        onLoadedMetadata={handleVideoMetadata}
      />

      <div style={containerStyle} className="relative overflow-hidden bg-black select-none">
        <video
          ref={bgVideoRef}
          className="absolute inset-0 w-full h-full object-cover scale-x-[-1] z-0"
          playsInline
          muted
        />

        {isReady && videoTexture && pointsState.length === 4 && (
          <div className="absolute inset-0 w-full h-full pointer-events-none z-10">
            <Canvas
              orthographic
              camera={{ left: -1, right: 1, top: 1, bottom: -1, near: 0.1, far: 10 }}
              style={{ width: '100%', height: '100%' }}
            >
              <XRayWindow
                pointsRef={pointsRef}
                videoTexture={videoTexture}
                videoWidth={videoWidth}
                videoHeight={videoHeight}
                effectMode={effectMode}
              />
            </Canvas>
          </div>
        )}

        <canvas
          ref={canvasRef}
          width={videoWidth}
          height={videoHeight}
          className="absolute inset-0 w-full h-full pointer-events-none z-20"
        />

        {isReady && pointsState.length === 4 && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-20">
            <defs>
              <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow
                  dx="0"
                  dy="0"
                  stdDeviation="6"
                  floodColor={effectMode === 'xray' ? '#00e5ff' : '#007fff'}
                  floodOpacity="0.8"
                />
              </filter>
            </defs>
            <polygon
              points={pointsState.map(p => `${(1.0 - p.x) * 100},${p.y * 100}`).map(coord => `${coord}%`).join(' ')}
              fill="none"
              stroke={effectMode === 'xray' ? '#00e5ff' : '#007fff'}
              strokeWidth="2"
              filter="url(#glow)"
            />
          </svg>
        )}

        {isReady && pointsState.length === 4 && pointsState.map((pt, idx) => (
          <div
            key={idx}
            className="absolute w-3 h-3 bg-[#00ff66] rounded-[2px] z-20 pointer-events-none shadow-[0_0_10px_#00ff66,0_0_20px_#00ff66]"
            style={{
              left: `${(1.0 - pt.x) * 100}%`,
              top: `${pt.y * 100}%`,
              transform: 'translate(-50%, -50%)'
            }}
          />
        ))}

        {!isReady && (
          <div className="absolute inset-0 w-full h-full flex flex-col items-center justify-center bg-black/85 text-white font-medium text-lg z-30 space-y-4 animate-pulse">
            <div className="w-12 h-12 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
            <div>Loading AI Models & Camera...</div>
          </div>
        )}
      </div>
    </div>
  );
}
