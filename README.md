# X-Ray Hand Tracker

An interactive WebGL and computer vision application that tracks hand landmarks in real-time, overlaying custom shader effects inside a deformable window.

## Features

- **MediaPipe Hand Landmark Tracking:** Identifies and draws the hand skeleton for up to two hands using GPU acceleration.
- **Deformable XRayWindow:** A 32x32 plane deformed dynamically using bilateral interpolation of coordinates from the tracked hands.
- **Interactive Shaders:**
  - **Particle Effect:** Twinkling, color-cycling grid particles over animated topographic contour lines.
  - **X-Ray Effect:** Real-time medical imaging visual effect featuring Sobel edge detection, film grain, and horizontal scanlines.
- **Gesture Mode Control:** Pinching both hands activates X-Ray Mode; releasing the pinch swaps back to Particle Mode.
- **Responsive Layout:** Dynamic resize listener locks a 16:9 webcam aspect ratio to prevent image distortion.

## Getting Started

### Prerequisites

- Node.js 18+
- pnpm (installed globally via `npm install -g pnpm`)

### Installation

```bash
pnpm install
```

### Development

```bash
pnpm run dev
```

### Production Build & Deploy

Build the client code and bundle the Express server:

```bash
pnpm run build
```

Run the production server:

```bash
pnpm start
```

## Technology Stack

- **Frontend:** React 19, Three.js, React Three Fiber (R3F), Tailwind CSS (v4), MediaPipe Tasks Vision
- **Backend:** Express, Node.js
- **Tooling:** Vite, TypeScript, esbuild
