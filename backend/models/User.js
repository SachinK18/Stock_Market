// Simple in-memory user storage (for development)
// In production, you would use a proper database like MongoDB, PostgreSQL, etc.

class User {
  constructor(id, name, email, password) {
    this.id = id;
    this.name = name;
    this.email = email;
    this.password = password;
    this.createdAt = new Date();
  }
}

// In-memory storage
let users = [];
let nextId = 1;

const UserModel = {
  // Create a new user
  create: (userData) => {
    const user = new User(
      nextId++,
      userData.name,
      userData.email,
      userData.password
    );
    users.push(user);
    return user;
  },

  // Find user by email
  findByEmail: (email) => {
    return users.find(user => user.email === email);
  },

  // Find user by ID
  findById: (id) => {
    return users.find(user => user.id === id);
  },

  // Get all users (for debugging)
  getAll: () => {
    return users;
  },

  // Check if email exists
  emailExists: (email) => {
    return users.some(user => user.email === email);
  }
};

module.exports = UserModel;
