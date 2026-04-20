/**
 * Build and download Bhavesh's contact card as a .vcf file (RFC 6350).
 * Opens in iOS / macOS Contacts, Google Contacts, Outlook — just tap the
 * downloaded file on any device and it'll offer to add to the address book.
 */

const VCARD = `BEGIN:VCARD
VERSION:3.0
FN:Bhavesh Gupta
N:Gupta;Bhavesh;;;
TITLE:MS Computer Science · NYU Courant
ORG:NYU Courant Institute;Computer Science
EMAIL;TYPE=INTERNET,PREF:bg2896@nyu.edu
TEL;TYPE=CELL:+12014928876
ADR;TYPE=HOME:;;Washington Square;Manhattan;NY;;USA
URL:https://libralpanda.vercel.app
URL;TYPE=linkedin:https://linkedin.com/in/bhaveshgupta01
URL;TYPE=github:https://github.com/bhaveshgupta01
NOTE:Building AI for problems that matter. Android · Backend · Agentic ML.
REV:2026-04-19T00:00:00Z
END:VCARD
`.replace(/\r?\n/g, '\r\n'); // vCard spec requires CRLF line endings

export function downloadVCard() {
  const blob = new Blob([VCARD], { type: 'text/vcard;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'Bhavesh_Gupta.vcf';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // release object URL after the browser has had a moment to handle it
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export { VCARD };
