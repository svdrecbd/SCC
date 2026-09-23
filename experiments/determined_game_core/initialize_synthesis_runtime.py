"""Unpack pinned Java packages privately, preserving acquisition evidence."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time


def main():
    directory = Path(sys.argv[1]).resolve()
    configuration = json.loads((directory / 'config.json').read_text())
    started = time.perf_counter()
    destination = directory / 'runtime'
    destination.mkdir()
    records = []
    for package in configuration['packages']:
        metadata = subprocess.check_output(['apt-cache', 'show', package['name'] + '=' + configuration['version']], text=True)
        (directory / (package['name'] + '.metadata.txt')).write_text(metadata)
        if 'SHA256: ' + package['sha256'] not in metadata:
            raise ValueError('Configured package hash differs from host package metadata.')
        subprocess.run(['apt-get', 'download', package['name'] + '=' + configuration['version']], cwd=directory, check=True, timeout=25)
        paths = list(directory.glob(package['name'] + '_*.deb'))
        if len(paths) != 1:
            raise ValueError('Expected exactly one downloaded package.')
        payload = paths[0].read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != package['sha256'] or len(payload) != package['bytes']:
            raise ValueError('Package checksum or length mismatch.')
        records.append({'package': package['name'], 'sha256': digest, 'bytes': len(payload)})
        del payload
        subprocess.run(['dpkg-deb', '-x', str(paths[0]), str(destination)], check=True)
    java_home = destination / 'usr/lib/jvm/java-17-openjdk-amd64'
    replaced = []
    for path in java_home.rglob('*'):
        if path.is_symlink():
            target = os.readlink(path)
            if target.startswith('/etc/java-17-openjdk/'):
                replacement = destination / target.lstrip('/')
                if not replacement.exists():
                    raise ValueError('Private Java configuration target missing.')
                path.unlink()
                path.symlink_to(os.path.relpath(replacement, path.parent))
                replaced.append(str(path.relative_to(java_home)))
    certificate_store = java_home / 'lib/security/cacerts'
    if certificate_store.is_symlink():
        certificate_store.unlink()
    source = directory / 'InitializeCertificateStore.java'
    source.write_text('''import java.io.*;
import java.security.KeyStore;
import java.security.cert.CertificateFactory;
public class InitializeCertificateStore {
    public static void main(String[] arguments) throws Exception {
        KeyStore store = KeyStore.getInstance("JKS");
        store.load(null, null);
        int count = 0;
        try (InputStream input = new FileInputStream(arguments[0])) {
            for (var certificate : CertificateFactory.getInstance("X.509").generateCertificates(input)) {
                store.setCertificateEntry("certificate_" + count++, certificate);
            }
        }
        try (OutputStream output = new FileOutputStream(arguments[1])) {
            store.store(output, "changeit".toCharArray());
        }
        System.out.println("Imported certificates: " + count);
    }
}
''')
    subprocess.run([str(java_home / 'bin/java'), str(source), '/etc/ssl/certs/ca-certificates.crt', str(certificate_store)], check=True, timeout=10)
    version = subprocess.run([str(java_home / 'bin/java'), '-version'], capture_output=True, text=True, check=True)
    result = {'packages': records, 'java_home': str(java_home), 'version': version.stderr,
              'private_links': replaced, 'certificate_store_sha256': hashlib.sha256(certificate_store.read_bytes()).hexdigest(),
              'seconds': time.perf_counter() - started, 'system_packages_changed': False, 'training': False}
    (directory / 'runtime_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
