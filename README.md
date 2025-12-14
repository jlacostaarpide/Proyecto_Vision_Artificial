# 🚀 Flujo de Trabajo Esencial con Ramas en Git

Este documento describe el flujo de trabajo estándar usando ramas (branches) para mantener el repositorio ordenado, permitir el trabajo en paralelo y asegurar que la rama principal (`main`) siempre esté estable.

---

## 1. Sincronizar Antes de Empezar

Antes de crear una nueva rama, asegúrate de que tu `main` local está perfectamente sincronizado con el repositorio remoto (GitHub).

```bash
# 1. Cambia a tu rama principal
git checkout main

# 2. Trae y aplica los últimos cambios del repositorio remoto
git pull
```

💻 **En GitHub Desktop:**
1. Abre **GitHub Desktop** y selecciona tu repositorio en el menú superior.
2. Asegúrate de que la rama seleccionada sea `main` (puedes verla en la parte superior del programa).
3. Haz clic en el botón **“Fetch origin”** para traer los últimos cambios.
4. Si aparece el botón **“Pull origin”**, haz clic para actualizar tu rama local.

---

## 2. Crear y Cambiar de Rama

Desde tu rama main actualizada, creas una "copia" nueva para desarrollar tu funcionalidad sin afectar a nadie más.

### Crear una Nueva Rama

Usa checkout -b para crear una nueva rama y cambiarte a ella en un solo paso.

```bash
# Crea la rama 'feature/nuevo-login' y te mueve a ella
git checkout -b feature/nuevo-login
```

> Buenas prácticas para nombres: Usa prefijos como `feature/` (nueva funcionalidad), `fix/` (corregir un error) o `docs/` (documentación).

💻 **En GitHub Desktop:**
1. Ve al menú superior y selecciona **“Branch → New Branch…”**.
2. Escribe un nombre como `feature/nuevo-login` y haz clic en **“Create Branch”**.
3. GitHub Desktop te cambiará automáticamente a esa nueva rama.

### Cambiar entre Ramas Existentes

Si la rama ya existe (o quieres volver a main), usas checkout sin el `-b`.

```bash
# Cambiar a una rama que ya existe
git checkout nombre-de-la-rama

# Volver a la rama principal
git checkout main
```

💻 **En GitHub Desktop:**
1. Haz clic en el menú desplegable de ramas (arriba, donde muestra el nombre actual).
2. Selecciona la rama a la que quieras cambiar.
3. Para volver a `main`, simplemente selecciónala desde la lista.

---

## 3. Trabajar en tu Rama (Tu Flujo Habitual)

Una vez en tu rama, trabajas con normalidad. Estos commit solo existen, por ahora, en esta rama.

```bash
# ...haces tus cambios en los archivos...

# Añades los archivos al "stage"
git add .

# Guardas los cambios con un mensaje
git commit -m "Mi primer avance en el login"

# ...trabajas más...
git add .
git commit -m "Terminada la validación del formulario"
```

💻 **En GitHub Desktop:**
1. Realiza tus cambios en el código desde tu editor (VSCode, etc.).
2. GitHub Desktop detectará automáticamente los archivos modificados en el panel izquierdo.
3. Marca las casillas de los archivos que quieres incluir en el commit.
4. Escribe un mensaje descriptivo en la parte inferior (“Summary”) y haz clic en **“Commit to <nombre-de-la-rama>”**.

---

## 4. Subir tu Rama a GitHub

Cuando quieras compartir tu progreso o tener una copia de seguridad, sube tu rama al repositorio remoto (origin).

### La Primera Vez que Subes la Rama

La primera vez debes "conectar" tu rama local con la remota usando `-u` (`--set-upstream`).

```bash
# Sube 'feature/nuevo-login' y la conecta con 'origin'
git push -u origin feature/nuevo-login
```

💻 **En GitHub Desktop:**
1. Una vez hecho tu commit, verás un botón azul **“Publish branch”** en la parte superior.
2. Haz clic allí para subir la rama al repositorio remoto.

### Siguientes Veces

Después de esa primera vez, si haces más commits en esta misma rama, solo necesitas:

```bash
git push
```

💻 **En GitHub Desktop:**
- Cada vez que hagas nuevos commits, simplemente haz clic en el botón **“Push origin”** (arriba).

---

## 5. Actualizar tu Rama con Main (IMPORTANTE)

Antes de entregar tu trabajo, es probable que tus compañeros hayan subido cosas nuevas a main. Debes meter esos cambios de main en tu rama para comprobar que tu código no rompe nada.

```bash
# 1. Asegúrate de estar en TU rama
git checkout mi-rama

# 2. Descarga la info de la nube (sin fusionar aún)
git fetch origin

# 3. Fusiona lo nuevo de main DENTRO de tu rama
git merge origin/main

# 4. Si hay conflictos, arréglalos, haz commit y sube el resultado
git push
```

💻 **En GitHub Desktop:**

1. Asegúrate de que en "Current Branch" estás en tu rama.
2. Haz clic en “Fetch origin” para actualizar datos.
3. Ve al menú superior: “Branch” → “Merge into current branch…”.
4. En la lista, selecciona main (debería decir algo como "Merge main into mi-rama").
5. Haz clic en el botón azul de confirmar.
6. Si todo sale bien, haz clic en “Push origin” para subir esa actualización.

¡Listo! Tu código ahora es parte oficial de la rama main.

---

## 6. El Pull Request (PR): Entregar el Trabajo

Ahora que tu rama está actualizada con lo último de main y probada, estás listo para integrarla oficialmente.

1. Ve a la página de tu repositorio en GitHub.
2. GitHub detectará tu rama y mostrará el botón "Compare & pull request".
3. Pon un título y descripción.
4. Verificación: GitHub te dirá "Able to merge" (porque ya resolviste los conflictos en el paso 5).
5. Un compañero (o tú) revisa y hace clic en "Merge pull request".

💻 **En GitHub Desktop:**
- Usa **“View on GitHub”** → “Create Pull Request”.

---

## 7. Limpieza

Una vez que tu PR ha sido fusionado en main, tu rama de trabajo ya no es necesaria y puede borrarse para mantener el repositorio limpio.

```bash
# 1. Vuelve a tu rama principal
git checkout main

# 2. Actualízala (para bajarte los cambios que acabas de fusionar)
git pull

# 3. Borra la rama local (ya no la necesitas)
git branch -d feature/nuevo-login
```

💻 **En GitHub Desktop:**
1. Cambia a la rama `main` desde el menú de ramas.
2. Haz clic en **“Fetch origin”** y luego **“Pull origin”** para actualizarla.
3. Ve al menú **“Branch → Delete…”** y selecciona la rama que ya fusionaste.
4. GitHub también te ofrecerá borrar la rama remota después del merge, directamente desde la página web del PR.

---

## 🧠 Resumen: La Chuleta Diaria

### Empezar una Tarea Nueva

```bash
git checkout main
git pull
git checkout -b mi-nueva-rama
```

💻 **En GitHub Desktop:**
1. Selecciona la rama `main`.
2. Haz clic en **“Fetch origin” → “Pull origin”**.
3. Ve a **“Branch → New Branch…”**, nómbrala y créala.

### Trabajar y Guardar Progreso

```bash
# ...editar archivos...
git add .
git commit -m "Mi progreso"
git push
```

💻 **En GitHub Desktop:**
1. Modifica tus archivos.
2. Revisa los cambios en el panel izquierdo.
3. Escribe un mensaje de commit y presiona **“Commit to <rama>”**.
4. Luego haz clic en **“Push origin”**.

> (Recuerda usar `git push -u origin mi-nueva-rama` la primera vez)

### Terminar y Proponer

1. Estando en tu rama: Branch → Merge into current branch... → Select main.
2. Resolver conflictos (si los hay) y hacer Push.
3. Ir a GitHub.
4. Crear Pull Request.
5. Esperar a que se revise y se fusione (Merge).

💻 **En GitHub Desktop:**
- Usa **“View on GitHub”** → “Create Pull Request”.

### Limpiar

```bash
git checkout main
git pull
git branch -d mi-nueva-rama
```

💻 **En GitHub Desktop:**
1. Cambia a `main` → “Fetch origin” → “Pull origin”.
2. Elimina la rama fusionada desde **Branch → Delete**.
