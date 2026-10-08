// SecureFinger Secure Enclave helper (macOS, Apple silicon or T2).
//
// Creates and uses a NON-EXPORTABLE ECDSA P-256 key inside the Secure Enclave via CryptoKit.
// The private key never leaves the enclave; the file written by `create` is an opaque,
// enclave-encrypted handle that is useless on any other device.
//
// Build (Xcode command-line tools):   swiftc -O tools/se_helper.swift -o tools/se_helper
//
// Usage:
//   se_helper check
//   se_helper create  <keyfile> [none|biometry|presence]   -> prints SubjectPublicKeyInfo DER, base64
//   se_helper pubkey  <keyfile>                            -> prints SubjectPublicKeyInfo DER, base64
//   se_helper sign    <keyfile> [reason]   (message on stdin) -> prints ECDSA DER signature, base64
//
// Modes:  none      key usable without user interaction (decision key)
//         biometry  the enclave refuses to sign until Touch ID succeeds (.biometryCurrentSet);
//                   enrolling a new finger in macOS invalidates the key
//         presence  Touch ID or the login password (.userPresence)
import CryptoKit
import Foundation
import LocalAuthentication
import Security

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    fail("usage: se_helper check | create <keyfile> [none|biometry|presence] | pubkey <keyfile> | sign <keyfile> [reason]")
}
guard SecureEnclave.isAvailable else { fail("Secure Enclave not available on this Mac", code: 3) }

switch args[1] {
case "check":
    print("ok")

case "create":
    guard args.count >= 3 else { fail("create needs <keyfile>") }
    let mode = args.count >= 4 ? args[3] : "none"
    var flags: SecAccessControlCreateFlags = [.privateKeyUsage]
    switch mode {
    case "none": break
    case "biometry": flags.insert(.biometryCurrentSet)
    case "presence": flags.insert(.userPresence)
    default: fail("unknown mode \(mode)")
    }
    var cfError: Unmanaged<CFError>?
    guard let access = SecAccessControlCreateWithFlags(
        kCFAllocatorDefault, kSecAttrAccessibleWhenUnlockedThisDeviceOnly, flags, &cfError)
    else { fail("access control: \(String(describing: cfError?.takeRetainedValue()))") }
    do {
        let key = try SecureEnclave.P256.Signing.PrivateKey(accessControl: access)
        try key.dataRepresentation.write(to: URL(fileURLWithPath: args[2]), options: .atomic)
        print(key.publicKey.derRepresentation.base64EncodedString())
    } catch {
        fail("create failed: \(error)", code: 4)
    }

case "pubkey":
    guard args.count >= 3 else { fail("pubkey needs <keyfile>") }
    do {
        let blob = try Data(contentsOf: URL(fileURLWithPath: args[2]))
        let key = try SecureEnclave.P256.Signing.PrivateKey(dataRepresentation: blob)
        print(key.publicKey.derRepresentation.base64EncodedString())
    } catch {
        fail("pubkey failed: \(error)", code: 4)
    }

case "sign":
    guard args.count >= 3 else { fail("sign needs <keyfile>") }
    let reason = args.count >= 4 ? args[3] : "approve a SecureFinger sign-in"
    let message = FileHandle.standardInput.readDataToEndOfFile()
    let context = LAContext()
    context.localizedReason = reason
    do {
        let blob = try Data(contentsOf: URL(fileURLWithPath: args[2]))
        let key = try SecureEnclave.P256.Signing.PrivateKey(dataRepresentation: blob, authenticationContext: context)
        let signature = try key.signature(for: message)   // ECDSA over SHA-256(message)
        print(signature.derRepresentation.base64EncodedString())
    } catch {
        fail("sign refused: \(error)", code: 5)
    }

default:
    fail("unknown command \(args[1])")
}
