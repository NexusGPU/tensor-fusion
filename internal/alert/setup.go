package alert

// offer API to install/update prometheus alertmanager with configMap and values from configuration with a single statefulSet
// let user to manage and upgrade alertmanager by themselves
// wrap notification configurations and change config map then trigger reload like prometheus operator does (if not install by tensor-fusion, let user to use AlertManagerConfig by themselves, tensor-fusion will only trigger alert to pre-configured alertmanager endpoint)

// use config map to manager alertmanager config

// TODO:
