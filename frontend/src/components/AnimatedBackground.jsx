// src/components/AnimatedBackground.jsx
import React from "react";
import styled, { keyframes } from "styled-components";

/* ---------- Animations ---------- */
const moveBlob = keyframes`
  0% { transform: translate(0, 0) scale(1); }
  25% { transform: translate(40vw, 20vh) scale(1.2); }
  50% { transform: translate(-35vw, 40vh) scale(0.9); }
  75% { transform: translate(30vw, -15vh) scale(1.1); }
  100% { transform: translate(0, 0) scale(1); }
`;

const rotateBlob = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

/* ---------- Styled Components ---------- */
const BackgroundContainer = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  z-index: 0; /* content must have higher z-index */
  background: #fffde7; /* light yellow-white background */
`;

const Blob = styled.div`
  position: absolute;
  border-radius: 50%;
  animation:
    ${moveBlob} ${(props) => props.duration}s infinite alternate ease-in-out,
    ${rotateBlob} ${(props) => props.rotateDuration}s infinite linear;
  background: radial-gradient(
    circle at center,
    ${(props) => props.color1} 0%,
    ${(props) => props.color2} 100%
  );
  width: ${(props) => props.size}px;
  height: ${(props) => props.size}px;
  top: ${(props) => props.top}%;
  left: ${(props) => props.left}%;
  transform-origin: center;
  box-shadow: 0 0 180px 60px ${(props) => props.color1};
  filter: blur(80px);
`;

/* ---------- Component ---------- */
const AnimatedBackground = () => {
  return (
    <BackgroundContainer>
      <Blob
        size={400}
        duration={18}
        rotateDuration={35}
        color1="#fffacd"
        color2="#ffffff"
        top={10}
        left={-10}
      />
      <Blob
        size={500}
        duration={22}
        rotateDuration={45}
        color1="#ffffe0"
        color2="#fffaf0"
        top={50}
        left={80}
      />
      <Blob
        size={300}
        duration={16}
        rotateDuration={30}
        color1="#f0e68c"
        color2="#fffff0"
        top={70}
        left={10}
      />
      <Blob
        size={450}
        duration={20}
        rotateDuration={40}
        color1="#fff8dc"
        color2="#fffff0"
        top={-5}
        left={40}
      />
      <Blob
        size={350}
        duration={24}
        rotateDuration={50}
        color1="#fafad2"
        color2="#ffffff"
        top={30}
        left={-30}
      />
    </BackgroundContainer>
  );
};

export default AnimatedBackground;
