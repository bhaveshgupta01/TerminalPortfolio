/**
 * Bhavesh's signature mark — a single continuous 6-point path traced from the
 * hand-drawn original. Uses `currentColor` so it inherits from whatever
 * `text-*` class the parent sets (works on light and dark themes).
 */
const LogoIcon = ({ className = '', strokeWidth = 45, ...rest }) => (
  <svg
    viewBox="0 0 1024 1024"
    className={className}
    xmlns="http://www.w3.org/2000/svg"
    role="img"
    aria-label="Bhavesh signature mark"
    {...rest}
  >
    <path
      d="M 280 860
         L 580 90
         L 170 490
         L 850 710
         L 390 890
         L 680 190"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

export default LogoIcon;
